import math
from typing import Iterable
import numpy as np
import pandas as pd
from psyki.ski import Injector
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model
from knowledge.utils import *
from results import PATH as RESULTS_PATH

UCI_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
SPLICE_JUNCTION_DATA_URL: str = UCI_URL + "molecular-biology/splice-junction-gene-sequences/splice.data"
BREAST_CANCER_DATA_URL: str = UCI_URL + "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
LOSS: str = "sparse_categorical_crossentropy"
OPTIMIZER: str = "adam"
METRICS: str = "accuracy"


def load_splice_junction_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(SPLICE_JUNCTION_DATA_URL, sep=",\s*", header=None, encoding='utf8')
    df.columns = ["class", "origin", "DNA"]

    def binarize_features(df_x: pd.DataFrame, mapping: dict[str: set[str]]) -> pd.DataFrame:
        def get_values() -> Iterable[str]:
            result = set()
            for values_set in mapping.values():
                for v in values_set:
                    result.add(v)
            return result
        sub_features = sorted(get_values())
        results = []
        for _, r in df_x.iterrows():
            row_result = []
            for value in r:
                positive_features = mapping[value]
                for feature in sub_features:
                    row_result.append(1 if feature in positive_features else 0)
            results.append(row_result)
        return pd.DataFrame(results, dtype=int)

    # Split the DNA sequence
    x = []
    for _, row in df.iterrows():
        label, _, features = row
        features = list(f for f in features.lower())
        features.append(label.lower())
        x.append(features)
    df = pd.DataFrame(x)
    class_mapping = SPLICE_JUNCTION_CLASS_MAPPING
    new_y = df.iloc[:, -1:].applymap(lambda y: class_mapping[y] if y in class_mapping.keys() else y)
    new_x = binarize_features(df.iloc[:, :-1], SPLICE_JUNCTION_AGGREGATE_FEATURE)
    new_y.columns = [new_x.shape[1]]
    df = new_x.join(new_y)
    features_indices = list(range(-30, 0)) + list(range(1, 31))
    df.columns = ['X' + (str(i) if i > 0 else "_" + str(abs(i))) + b for i in features_indices for b in SPLICE_JUNCTION_FEATURES] + ['class']
    return df


def load_breast_cancer_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(BREAST_CANCER_DATA_URL, sep=",", header=None, encoding='utf8').iloc[:, 1:]
    df.columns = ["ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
                  "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "class"]
    return df


def create_neural_network(input_shape, neurons_per_layer) -> Model:
    net_input = Input(input_shape)
    x = Dense(neurons_per_layer[0], activation="relu")(net_input)
    for neurons in neurons_per_layer[1:-1]:
        x = Dense(neurons, activation="relu")(x)
    x = Dense(neurons_per_layer[-1], activation="softmax")(x)
    return Model(net_input, x)


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


def k_fold_cross_validation(dataset: pd.DataFrame, net: Model, kb: list[Formula], k: int = 10, train_size: int = 1000,
                            population: int = 30, seed: int = 0, epochs: int = 100, batch_size: int = 32):

    def compile_nets(nets: Iterable[Model]):
        for n in nets:
            n.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    def train_nets(nets: Iterable[Model], x, y, callbacks):
        for n in nets:
            n.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)

    def make_predictions_with_k_nets(nets: list[Model], x) -> np.ndarray:
        y_pred = []
        for n in nets:
            y_pred.append(n.predict(x))
        return np.asarray(np.argmax(sum(y_pred), axis=1))

    feature_mapping = {f: i for i, f in enumerate(dataset.columns[:-1])}
    kins = Injector.kins(net, feature_mapping)
    kbann = Injector.kbann(net, feature_mapping, omega=1)
    for p in range(population):
        print(f"\nPopulation {p+1} out of {population}")
        set_seed(seed + p)
        vanilla_nets, kins_nets, kbann_nets = [], [], []
        train, test = train_test_split(dataset, train_size=train_size, random_state=seed, stratify=dataset.iloc[:, -1])
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, _ in k_fold.split(train.iloc[:, :-1], train.iloc[:, -1:]):
            train_x, train_y = train.iloc[train_idx, :-1], train.iloc[train_idx, -1:]
            early_stop = Conditions(train_x, train_y)
            vanilla_net = clone_model(net)
            kins_net = kins.inject(kb)
            kbann_net = kbann.inject(kb)
            compile_nets([vanilla_net, kins_net, kbann_net])
            train_nets([vanilla_net, kins_net, kbann_net], train_x, train_y, early_stop)
            vanilla_nets.append(vanilla_net)
            kins_nets.append(kins_net)
            kbann_nets.append(kbann_net)
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
        vanilla_pred = make_predictions_with_k_nets(vanilla_nets, test_x)
        kins_pred = make_predictions_with_k_nets(kins_nets, test_x)
        kbann_pred = make_predictions_with_k_nets(kbann_nets, test_x)
        results = pd.DataFrame([test_y.to_numpy(), vanilla_pred, kins_pred, kbann_pred]).T
        results.columns = ["y_true", "y_vanilla", "y_kins", "y_kbann"]
        results.to_csv(RESULTS_PATH / ("vanilla_p" + str(p + 1) + "_s" + str(seed) + ".csv"), index=False)


def get_neurons_per_layer(input_shape, output_shape) -> list[int]:
    neurons_per_layer = []
    depth = math.floor(math.log10(input_shape))
    first_hidden_layer_neurons = math.ceil(math.log2(input_shape))**2
    neurons_per_layer.append(first_hidden_layer_neurons)
    last_layer_neurons = first_hidden_layer_neurons
    for _ in range(depth-1):
        last_layer_neurons /= 2
        neurons_per_layer.append(int(last_layer_neurons))
    neurons_per_layer.append(output_shape)
    return neurons_per_layer


def run_experiments(dataset_name: str, population: int, seed: int):
    if dataset_name == 's':  # s stands for splice junction
        dataset: pd.DataFrame = load_splice_junction_dataset()
        kb: list[Formula] = load_splice_junction_knowledge()
        output_shape: int = len(SPLICE_JUNCTION_CLASS_MAPPING.keys())
        train_size: int = 1000
    else:
        dataset: pd.DataFrame = load_breast_cancer_dataset()
        kb: list[Formula] = load_breast_cancer_knowledge()
        output_shape: int = 2
        train_size: int = 250
    net = create_neural_network(dataset.shape[1] - 1, get_neurons_per_layer(dataset.shape[1] - 1, output_shape))
    net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    k_fold_cross_validation(dataset, net, kb, train_size=train_size, seed=seed)

