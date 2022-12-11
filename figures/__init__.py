from pathlib import Path
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from dataset import SPLICE_JUNCTION_CLASS_MAPPING, BREAST_CANCER_CLASS_MAPPING_SHORT
from statistics import PATH as STATISTICS_PATH


PATH = Path(__file__).parents[0]

# Knowledge is provided in the dataset and this is the corresponding confusion matrix
SPLICE_JUNCTION_RULES_CONFUSION_MATRIX = np.array([[295, 0, 473], [25, 31, 711], [3, 0, 1652]])


def get_breast_cancer_confusion_matrix():
    # Knowledge is generated, therefore this method reads the corresponding performance file
    df = pd.read_csv(STATISTICS_PATH / "breast-cancer-rules.csv")
    return np.array([df.iloc[0, 1:], df.iloc[1, 1:]]).astype(int)


def confusion_matrix_figure(name: str = 's'):
    data = SPLICE_JUNCTION_RULES_CONFUSION_MATRIX if name == 's' else get_breast_cancer_confusion_matrix()
    class_names = SPLICE_JUNCTION_CLASS_MAPPING.keys() if name == 's' else BREAST_CANCER_CLASS_MAPPING_SHORT.keys()
    fig, ax = plot_confusion_matrix(data, show_absolute=True, show_normed=True, colorbar=True, class_names=class_names)
    fig.savefig(PATH / (name + '-confusion-matrix.pdf'), transparent=True)
