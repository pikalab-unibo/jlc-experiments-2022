import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from results import PATH as RESULTS_PATH

PATH = Path(__file__).parents[0]


def true_positive(y_true, y_pred, value):
    return np.sum((y_true == value) * (y_pred == value))


def true_negative(y_true, y_pred, value):
    return np.sum((y_true != value) * (y_pred != value))


def false_positive(y_true, y_pred, value):
    return np.sum((y_true != value) * (y_pred == value))


def false_negative(y_true, y_pred, value):
    return np.sum((y_true == value) * (y_pred != value))


def precision(y_true, y_pred, value):
    tp = true_positive(y_true, y_pred, value)
    return 0 if tp == 0 else tp / (tp + false_positive(y_true, y_pred, value))


def recall(y_true, y_pred, value):
    tp = true_positive(y_true, y_pred, value)
    return tp / (tp + false_negative(y_true, y_pred, value))


def f1(y_true, y_pred, value):
    p = precision(y_true, y_pred, value)
    r = recall(y_true, y_pred, value)
    return 0 if p == 0 or r == 0 else 2 * ((p * r) / (p + r))


def accuracy(y_true, y_pred, value):
    if isinstance(value, list):
        return np.sum([true_positive(y_true, y_pred, v) for v in value]) / y_true.shape[0]
    else:
        return (true_positive(y_true, y_pred, value) + true_negative(y_true, y_pred, value)) / y_true.shape[0]


def compute_statistics(population: int = 30):
    sj_vanilla, sj_kins, sj_kbann, sj_dt, sj_knn3, sj_knn5 = [], [], [], [], [], []
    bc_vanilla, bc_kins, bc_kbann, bc_dt, bc_knn3, bc_knn5 = [], [], [], [], [], []
    for p in range(population):
        # Splice junction
        def compute_sj_stats(data, index):
            return np.mean([[accuracy(r.y_true, r.iloc[:, index], [0, 1, 2]),
                             accuracy(r.y_true, r.iloc[:, index], 0),
                             accuracy(r.y_true, r.iloc[:, index], 1),
                             accuracy(r.y_true, r.iloc[:, index], 2),
                             precision(r.y_true, r.iloc[:, index], 0),
                             precision(r.y_true, r.iloc[:, index], 1),
                             precision(r.y_true, r.iloc[:, index], 2),
                             recall(r.y_true, r.iloc[:, index], 0),
                             recall(r.y_true, r.iloc[:, index], 1),
                             recall(r.y_true, r.iloc[:, index], 2),
                             f1(r.y_true, r.iloc[:, index], 0),
                             f1(r.y_true, r.iloc[:, index], 1),
                             f1(r.y_true, r.iloc[:, index], 2)
                             ] for r in data], axis=0)
        sj_result_files = glob.glob(str(RESULTS_PATH / ("s_p" + str(p + 1) + "_*.csv")))
        sj_result_dfs = [pd.read_csv(file) for file in sj_result_files]
        sj_vanilla.append(compute_sj_stats(sj_result_dfs, 1))
        sj_kins.append(compute_sj_stats(sj_result_dfs, 2))
        sj_kbann.append(compute_sj_stats(sj_result_dfs, 3))
        sj_dt.append(compute_sj_stats(sj_result_dfs, 4))
        sj_knn3.append(compute_sj_stats(sj_result_dfs, 5))
        sj_knn5.append(compute_sj_stats(sj_result_dfs, 6))

        # Breast cancer
        def compute_bc_stats(data, index):
            return np.mean([[accuracy(r.y_true, r.iloc[:, index], [0, 1]),
                             precision(r.y_true, r.iloc[:, index], 0),
                             precision(r.y_true, r.iloc[:, index], 1),
                             recall(r.y_true, r.iloc[:, index], 0),
                             recall(r.y_true, r.iloc[:, index], 1),
                             f1(r.y_true, r.iloc[:, index], 0),
                             f1(r.y_true, r.iloc[:, index], 1),
                             ] for r in data], axis=0)

        bc_result_files = glob.glob(str(RESULTS_PATH / ("b_p" + str(p + 1) + "_*.csv")))
        bc_result_dfs = [pd.read_csv(file) for file in bc_result_files]
        bc_vanilla.append(compute_bc_stats(bc_result_dfs, 1))
        bc_kins.append(compute_bc_stats(bc_result_dfs, 2))
        bc_kbann.append(compute_bc_stats(bc_result_dfs, 3))
        bc_dt.append(compute_bc_stats(bc_result_dfs, 4))
        bc_knn3.append(compute_bc_stats(bc_result_dfs, 5))
        bc_knn5.append(compute_bc_stats(bc_result_dfs, 6))

    predictor_names = [r'\kins', 'NN (no knowledge)', r'\kbann', 'CART', 'K-NN (3)', 'K-NN (5)']
    sj_vanilla = pd.DataFrame(sj_vanilla + [np.mean(sj_vanilla, axis=0)])
    sj_kins = pd.DataFrame(sj_kins + [np.mean(sj_kins, axis=0)])
    sj_kbann = pd.DataFrame(sj_kbann + [np.mean(sj_kbann, axis=0)])
    sj_dt = pd.DataFrame(sj_dt + [np.mean(sj_dt, axis=0)])
    sj_knn3 = pd.DataFrame(sj_knn3 + [np.mean(sj_knn3, axis=0)])
    sj_knn5 = pd.DataFrame(sj_knn5 + [np.mean(sj_knn5, axis=0)])
    sj_vanilla.to_csv(PATH / "s_vanilla.csv")
    sj_kins.to_csv(PATH / "s_kins.csv")
    sj_kbann.to_csv(PATH / "s_kbann.csv")
    sj_dt.to_csv(PATH / "s_dt.csv")
    sj_knn3.to_csv(PATH / "s_knn3.csv")
    sj_knn5.to_csv(PATH / "s_knn5.csv")
    sj_stats_to_latex(predictor_names, [sj_kins, sj_vanilla, sj_kbann, sj_dt, sj_knn3, sj_knn5])
    s, p = ttest_ind(sj_vanilla.iloc[:, 0], sj_kins.iloc[:, 0])
    print(s)
    print(p)

    bc_vanilla = pd.DataFrame(bc_vanilla + [np.mean(bc_vanilla, axis=0)])
    bc_kins = pd.DataFrame(bc_kins + [np.mean(bc_kins, axis=0)])
    bc_kbann = pd.DataFrame(bc_kbann + [np.mean(bc_kbann, axis=0)])
    bc_dt = pd.DataFrame(bc_dt + [np.mean(bc_dt, axis=0)])
    bc_knn3 = pd.DataFrame(bc_knn3 + [np.mean(bc_knn3, axis=0)])
    bc_knn5 = pd.DataFrame(bc_knn5 + [np.mean(bc_knn5, axis=0)])
    bc_vanilla.to_csv(PATH / "b_vanilla.csv")
    bc_kins.to_csv(PATH / "b_kins.csv")
    bc_kbann.to_csv(PATH / "b_kbann.csv")
    bc_dt.to_csv(PATH / "b_dt.csv")
    bc_knn3.to_csv(PATH / "b_knn3.csv")
    bc_knn5.to_csv(PATH / "b_knn5.csv")
    bc_stats_to_latex(predictor_names, [bc_kins, bc_vanilla, bc_kbann, bc_dt, bc_knn3, bc_knn5])
    s, p = ttest_ind(bc_vanilla.iloc[:, 0], bc_kins.iloc[:, 0])
    print(s)
    print(p)


def sj_stats_to_latex(predictor_names, dfs):
    def r3(number):
        return str(round(number, 3))
    results = ''
    for name, df in zip(predictor_names, dfs):
        results += '\multirow{3}{*}{' + name + '} & \multirow{3}{*}{' + str(round(df.iloc[-1, 0], 3)) + '} & '
        results += ' & '.join([r3(df.iloc[-1, 1]), r3(df.iloc[-1, 4]), r3(df.iloc[-1, 7]), r3(df.iloc[-1, 10])])
        results += r' & $\const{exon-intron}$'
        results += '\n' + r'\\ & &'
        results += ' & '.join([r3(df.iloc[-1, 2]), r3(df.iloc[-1, 5]), r3(df.iloc[-1, 8]), r3(df.iloc[-1, 11])])
        results += r' & $\const{intron-exon}$'
        results += '\n' + r'\\ & &'
        results += ' & '.join([r3(df.iloc[-1, 3]), r3(df.iloc[-1, 6]), r3(df.iloc[-1, 9]), r3(df.iloc[-1, 12])])
        results += r' & $\const{none}$'
        results += '\n' + r'\\' + '\n' + r'\hline' + '\n'
    with open(PATH / "s_stats.tex", "w") as text_file:
        text_file.write(results)


def bc_stats_to_latex(predictor_names, dfs):
    def r3(number):
        return str(round(number, 3))
    results = ''
    for name, df in zip(predictor_names, dfs):
        results += '\multirow{2}{*}{' + name + '} & \multirow{2}{*}{' + str(round(df.iloc[-1, 0], 3)) + '} & '
        results += ' & '.join([r3(df.iloc[-1, 1]), r3(df.iloc[-1, 3]), r3(df.iloc[-1, 5])])
        results += r' & $\const{benign}$'
        results += '\n' + r'\\ & &'
        results += ' & '.join([r3(df.iloc[-1, 2]), r3(df.iloc[-1, 4]), r3(df.iloc[-1, 6])])
        results += r' & $\const{malignant}$'
        results += '\n' + r'\\' + '\n' + r'\hline' + '\n'
    with open(PATH / "b_stats.tex", "w") as text_file:
        text_file.write(results)


