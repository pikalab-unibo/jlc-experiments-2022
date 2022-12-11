import glob
from pathlib import Path
import numpy as np
import pandas as pd
from dataset import SPLICE_JUNCTION_CLASS_MAPPING, BREAST_CANCER_CLASS_MAPPING
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
    return tp / (tp + false_positive(y_true, y_pred, value))


def recall(y_true, y_pred, value):
    tp = true_positive(y_true, y_pred, value)
    return tp / (tp + false_negative(y_true, y_pred, value))


def f1(y_true, y_pred, value):
    p = precision(y_true, y_pred, value)
    r = recall(y_true, y_pred, value)
    return 2 * ((p * r) / (p + r))


def accuracy(y_true, y_pred, value):
    if isinstance(value, list):
        return np.sum([true_positive(y_true, y_pred, v) for v in value]) / y_true.shape[0]
    else:
        return (true_positive(y_true, y_pred, value) + true_negative(y_true, y_pred, value)) / y_true.shape[0]


def compute_statistics(population: int = 30):
    sj_vanilla, sj_kins, sj_kbann = [], [], []
    bc_vanilla, bc_kins, bc_kbann = [], [], []
    for p in range(population):
        # Splice junction
        sj_result_files = glob.glob(str(RESULTS_PATH / ("s_p" + str(p+1) + "*.csv")))
        sj_result_dfs = [pd.read_csv(file) for file in sj_result_files]
        sj_classes = SPLICE_JUNCTION_CLASS_MAPPING
        sj_vanilla.append(np.mean([[accuracy(r.y_true, r.iloc[:, 1], c) for c in sj_classes.values()] + [
            accuracy(r.y_true, r.iloc[:, 1], [0, 1, 2])] for r in sj_result_dfs], axis=0))
        sj_kins.append(np.mean([[accuracy(r.y_true, r.iloc[:, 2], c) for c in sj_classes.values()] + [
            accuracy(r.y_true, r.iloc[:, 2], [0, 1, 2])] for r in sj_result_dfs], axis=0))
        sj_kbann.append(np.mean([[accuracy(r.y_true, r.iloc[:, 3], c) for c in sj_classes.values()] + [
            accuracy(r.y_true, r.iloc[:, 3], [0, 1, 2])] for r in sj_result_dfs], axis=0))
        # Breast cancer
        bc_result_files = glob.glob(str(RESULTS_PATH / ("b_p" + str(p + 1) + "*.csv")))
        bc_result_dfs = [pd.read_csv(file) for file in bc_result_files]
        bc_classes = BREAST_CANCER_CLASS_MAPPING
        bc_vanilla.append(np.mean([[accuracy(r.y_true, r.iloc[:, 1], c) for c in bc_classes.values()] + [
            accuracy(r.y_true, r.iloc[:, 1], [0, 1])] for r in bc_result_dfs], axis=0))
        bc_kins.append(np.mean([[accuracy(r.y_true, r.iloc[:, 2], c) for c in bc_classes.values()] + [
            accuracy(r.y_true, r.iloc[:, 2], [0, 1])] for r in bc_result_dfs], axis=0))
        bc_kbann.append(np.mean([[accuracy(r.y_true, r.iloc[:, 3], c) for c in bc_classes.values()] + [
            accuracy(r.y_true, r.iloc[:, 3], [0, 1])] for r in bc_result_dfs], axis=0))

    sj_vanilla = pd.DataFrame(sj_vanilla + [np.mean(sj_vanilla, axis=0)])
    sj_kins = pd.DataFrame(sj_kins + [np.mean(sj_kins, axis=0)])
    sj_kbann = pd.DataFrame(sj_kbann + [np.mean(sj_kbann, axis=0)])
    sj_vanilla.to_csv(PATH / "s_vanilla.csv")
    sj_kins.to_csv(PATH / "s_kins.csv")
    sj_kbann.to_csv(PATH / "s_kbann.csv")
    bc_vanilla = pd.DataFrame(bc_vanilla + [np.mean(bc_vanilla, axis=0)])
    bc_kins = pd.DataFrame(bc_kins + [np.mean(bc_kins, axis=0)])
    bc_kbann = pd.DataFrame(bc_kbann + [np.mean(bc_kbann, axis=0)])
    bc_vanilla.to_csv(PATH / "b_vanilla.csv")
    bc_kins.to_csv(PATH / "b_kins.csv")
    bc_kbann.to_csv(PATH / "b_kbann.csv")