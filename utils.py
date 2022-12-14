import copy
import os.path
import re
from typing import Iterable
import numpy as np
import pandas as pd
from psyke import Extractor
from psyke.utils.logic import data_to_struct, pretty_theory
from psyki.logic import Formula
from psyki.ski import Injector
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model
from tuprolog.solve.prolog import prolog_solver
from tuprolog.theory import mutable_theory
from dataset import load_splice_junction_dataset, load_breast_cancer_dataset, SPLICE_JUNCTION_CLASS_MAPPING_SHORT
from knowledge import load_splice_junction_knowledge, load_breast_cancer_knowledge, PATH
from results import PATH as RESULTS_PATH
from statistics import true_positive, true_negative, false_positive, false_negative, PATH as STATISTICS_PATH

LOSS: str = "sparse_categorical_crossentropy"
OPTIMIZER: str = "adam"
METRICS: str = "accuracy"
SEED: int = 0
EPOCHS: int = 100
BATCH_SIZE: int = 32
SPLICE_JUNCTION_OMEGA: float = 4
BREAST_CANCER_OMEGA: float = 1


def create_neural_network(input_shape, neurons_per_layer) -> Model:
    net_input = Input(input_shape)
    x = Dense(neurons_per_layer[0], activation="relu")(net_input)
    for neurons in neurons_per_layer[1:-1]:
        x = Dense(neurons, activation="relu")(x)
    x = Dense(neurons_per_layer[-1], activation="softmax")(x)
    return Model(net_input, x)


def get_neurons_per_layer(input_shape, output_shape) -> list[int]:
    return [int(input_shape/6), output_shape]


class Conditions(Callback):
    def __init__(self, train_x, train_y, patience: int = 5, threshold: float = 0.25, stop_threshold_1: float = 0.99,
                 stop_threshold_2: float = 0.9):
        super(Conditions, self).__init__()
        self.train_x = train_x
        train_y = train_y.iloc[:, 0]
        self.train_y = np.zeros((train_y.size, train_y.max() + 1))
        self.train_y[np.arange(train_y.size), train_y] = 1
        self.patience = patience
        self.threshold = threshold
        self.stop_threshold_1 = stop_threshold_1
        self.stop_threshold_2 = stop_threshold_2
        self.best_acc = 0
        self.wait = 0
        self.best_weights = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        self.best_acc = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Second condition
        acc = logs.get('accuracy')
        if self.best_acc > acc > self.stop_threshold_2:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
        else:
            self.best_acc = acc
            self.wait = 0

            predictions = self.model.predict(self.train_x)
            errors = np.abs(predictions - self.train_y) <= self.threshold
            errors = np.sum(errors, axis=1)
            errors = len(errors[errors == predictions.shape[1]])
            is_over_threshold = errors / predictions.shape[0] > self.stop_threshold_1

            if is_over_threshold:
                self.best_weights = self.model.get_weights()
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def k_fold_cross_validation(name: str, dataset: pd.DataFrame, net: Model, kb: list[Formula], k: int = 10,
                            population: int = 30, seed: int = SEED, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):

    def compile_nets(nets: Iterable[Model]):
        for n in nets:
            n.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    def fit_predictors(predictors: Iterable[any], x, y, callbacks=None):
        for p in predictors:
            if isinstance(p, Model):
                p.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
            else:
                p.fit(x, y.iloc[:, 0])

    predictor_name_predictions = ['y_vanilla', 'y_kins', 'y_kbann', 'y_dt', 'y_knn3', 'y_knn5']
    # Injectors
    feature_mapping = {f: i for i, f in enumerate(dataset.columns[:-1])}
    kins = Injector.kins(net, feature_mapping)
    omega = SPLICE_JUNCTION_OMEGA if name == 's' else BREAST_CANCER_OMEGA
    kbann = Injector.kbann(net, feature_mapping, omega=omega, gamma=0)
    df = dataset.copy()
    for p in range(population):
        print(f"\nPopulation {p + 1} out of {population}")
        set_seed(seed + p)
        train = df.sample(frac=1, random_state=seed + p)
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + p)
        i: int = 0
        for train_idx, valid_idx in k_fold.split(train.iloc[:, :-1], train.iloc[:, -1:]):
            i += 1
            # Dataset split
            file = RESULTS_PATH / (name + "_p" + str(p+1) + "_k" + str(i) + "_s" + str(seed) + ".csv")
            train_x, train_y = train.iloc[train_idx, :-1], train.iloc[train_idx, -1:]
            valid_x, valid_y = train.iloc[valid_idx, :-1], train.iloc[valid_idx, -1]
            if name == 's':
                early_stop = Conditions(train_x, train_y)
            else:
                early_stop = Conditions(train_x, train_y, stop_threshold_2=0.95)
            # Predictors creation
            vanilla_net = clone_model(net)
            kins_net = kins.inject(kb)
            kbann_net = kbann.inject(kb)
            dt = DecisionTreeClassifier()
            knn3 = KNeighborsClassifier(n_neighbors=3)
            knn5 = KNeighborsClassifier(n_neighbors=5)
            predictors = [vanilla_net, kins_net, kbann_net, dt, knn3, knn5]
            predictors_map = {k: v for k, v in zip(predictor_name_predictions, predictors)}
            # remove predictor from training if results are already available
            results = {}
            if os.path.isfile(file):
                old_results = pd.read_csv(file)
                for predictions_name in old_results.columns[1:]:
                    if predictions_name in predictor_name_predictions and predictions_name not in ('y_kbann', 'y_kins', 'y_vanilla'):
                        results[predictions_name] = old_results.loc[:, predictions_name]
                        predictors_map.pop(predictions_name)
            compile_nets([vanilla_net, kins_net, kbann_net])
            fit_predictors(list(predictors_map.values()), train_x, train_y, early_stop)
            for n, predictor in predictors_map.items():
                if isinstance(predictor, Model):
                    results[n] = np.asarray(np.argmax(predictor.predict(valid_x), axis=1))
                else:
                    results[n] = np.asarray(predictor.predict(valid_x))
            r = list([results["y_vanilla"], results['y_kins'], results['y_kbann'], results['y_dt'], results['y_knn3'], results['y_knn5']])
            df_results = pd.DataFrame([valid_y.to_numpy()] + r).T
            df_results.columns = ["y_true"] + predictor_name_predictions
            df_results.to_csv(file, index=False)
            del kbann_net, kins_net, vanilla_net, dt, knn3, knn5


def run_experiments(dataset_name: str, population: int, seed: int):
    if dataset_name == 's':  # s stands for splice junction
        dataset: pd.DataFrame = load_splice_junction_dataset()
        kb: list[Formula] = load_splice_junction_knowledge()
        output_shape: int = len(SPLICE_JUNCTION_CLASS_MAPPING_SHORT.keys())
        net = create_neural_network(dataset.shape[1] - 1, [(dataset.shape[1] - 1)/6, output_shape])
    else:
        dataset: pd.DataFrame = load_breast_cancer_dataset()
        kb: list[Formula] = load_breast_cancer_knowledge()
        output_shape: int = 2
        net = create_neural_network(dataset.shape[1] - 1, [(dataset.shape[1] - 1)/5, output_shape])

    # net.summary()
    net.compile(optimizer=OPTIMIZER, loss=LOSS, metrics="accuracy")
    k_fold_cross_validation(dataset_name, dataset, net, kb, seed=seed, population=population)


def extract_knowledge(seed: int = 0, train_size: float = 200):
    dataset: pd.DataFrame = load_breast_cancer_dataset()
    set_seed(seed)
    predictor = DecisionTreeClassifier()
    train, _ = train_test_split(dataset, train_size=train_size, random_state=seed, stratify=dataset.iloc[:, -1])
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
    predictor.fit(train_x, train_y)
    cart = Extractor.cart(predictor, max_depth=None, max_leaves=None, simplify=False)
    knowledge = cart.extract(train, sort=False)
    solver = prolog_solver(static_kb=mutable_theory(knowledge))
    substitutions = [solver.solveOnce(data_to_struct(data.astype(int))) for _, data in dataset.iterrows()]
    y_pred = [str(query.solved_query.get_arg_at(dataset.shape[1] - 1)) for query in substitutions if query.is_yes]
    y_pred = [0 if y == "0.0" else 1 for y in y_pred]
    rule_results = pd.DataFrame([dataset.diagnosis.to_list(), y_pred]).T
    rule_results.columns = ['y_true', 'y_rules']
    tp = true_positive(rule_results.y_true, rule_results.y_rules, 0)
    tn = true_negative(rule_results.y_true, rule_results.y_rules, 0)
    fp = false_positive(rule_results.y_true, rule_results.y_rules, 0)
    fn = false_negative(rule_results.y_true, rule_results.y_rules, 0)
    matrix = pd.DataFrame([[tp, fn], [fp, tn]])
    matrix.columns = ["predicted benign", "predicted malignant"]
    matrix.index = ["true benign", "true malignant"]
    matrix.to_csv(STATISTICS_PATH / "breast-cancer-rules.csv")
    textual_knowledge = pretty_theory(knowledge)
    textual_knowledge = re.sub(r"([A-Z][a-zA-Z0-9]*)[ ]>[ ]([+-]?([0-9]*))[.]?[0-9]+", r"\g<1>", textual_knowledge)
    textual_knowledge = re.sub(r"([A-Z][a-zA-Z0-9]*)[ ]=<[ ]([+-]?([0-9]*))[.]?[0-9]+", r"not(\g<1>)",
                               textual_knowledge)
    textual_knowledge = re.sub(r"(diagnosis)\((.*, )('1')\)", r"class(\g<2>malignant)", textual_knowledge)
    textual_knowledge = re.sub(r"(diagnosis)\((.*, )('0')\)", r"class(\g<2>benign)", textual_knowledge)
    textual_knowledge_to_latex(textual_knowledge)
    with open(PATH / "breast-cancer-kb.pl", "w") as text_file:
        text_file.write(textual_knowledge)


def textual_knowledge_to_latex(knowledge: str):
    k = copy.copy(knowledge)
    results = {0: "", 1: ""}
    classes: dict[int: str] = {0: 'benign', 1: 'malignant'}
    for row in k.split(".\n"):
        index = 0 if '0.0' in row else 1
        partial_result = r'\begin{aligned}' + '\n' + r'\pred{class}(\bar{\var{X}}, \const{' + classes[index] + r'}) \leftarrow '
        _, variables = row.split(':-\n')
        variables = variables.replace(" ", "")
        variables = re.sub(r"([A-Z][a-zA-Z]*)([0-9]*)", r"\\var{\g<1>} = \\const{\g<2>}", variables)
        partial_result += r" \wedge ".join(r"\neg(" + v[4:-1] + r")" if "not" in v else v for v in variables.split(','))
        partial_result += "\n" + r'\end{aligned}' + '\n' + r"\\" + "\n"
        results[index] = results[index] + partial_result
    with open(PATH / "breast-cancer-kb-latex.tex", "w") as text_file:
        text_file.write(results[0])
        text_file.write(results[1])
