from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import boxplot
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


def accuracy_bar_plots(name: str = 's'):
    df_vanilla = pd.read_csv(STATISTICS_PATH / (name + "_vanilla.csv"), index_col=False)
    df_kins = pd.read_csv(STATISTICS_PATH / (name + "_kins.csv"))
    df_kbann = pd.read_csv(STATISTICS_PATH / (name + "_kbann.csv"))
    if name == 's':
        plt.figure(figsize=(12, 8))
        title = 'Splice junction per class accuracy'
        length = 3
        vanilla, kins, kbann = df_vanilla.iloc[:, 1:-1], df_kins.iloc[:, 1:-1], df_kbann.iloc[:, 1:-1]
    else:
        plt.figure(figsize=(6, 8))
        title = 'Breast Cancer overall accuracy'
        length = 1
        vanilla, kins, kbann = df_vanilla.iloc[:, -1], df_kins.iloc[:, -1], df_kbann.iloc[:, -1]
    main_color, border_color = 'blue', 'royalblue'
    box_vanilla = boxplot(np.array(vanilla), positions=list(np.arange(0.33, length + .33, 1)),
                          notch=False, patch_artist=True,
                          boxprops=dict(facecolor=border_color, color=main_color),
                          capprops=dict(color=main_color),
                          whiskerprops=dict(color=main_color),
                          flierprops=dict(color=main_color, markeredgecolor=main_color),
                          medianprops=dict(color=main_color), )
    main_color, border_color = 'lightgreen', 'darkgreen'
    box_kins = boxplot(np.array(kins), positions=list(np.arange(0.66, length + .66, 1)),
                       notch=False, patch_artist=True,
                       boxprops=dict(facecolor=border_color, color=main_color),
                       capprops=dict(color=main_color),
                       whiskerprops=dict(color=main_color),
                       flierprops=dict(color=main_color, markeredgecolor=main_color),
                       medianprops=dict(color=main_color), )
    main_color, border_color = 'red', 'pink'
    box_kbann = boxplot(np.array(kbann), positions=list(np.arange(1, length + 1, 1)),
                        notch=False, patch_artist=True,
                        boxprops=dict(facecolor=border_color, color=main_color),
                        capprops=dict(color=main_color),
                        whiskerprops=dict(color=main_color),
                        flierprops=dict(color=main_color, markeredgecolor=main_color),
                        medianprops=dict(color=main_color), )
    plt.ylabel('Accuracy', fontsize=20)
    plt.title(title, fontsize=24)
    predictors = ["vanilla NN", "KINS", "KBANN"]
    if name == 's':
        plt.xticks(list(np.arange(0.66, length + .66, 1)), list(np.arange(0.66, length + .66, 1)))
        plt.xlabel('Classes', fontsize=20)
        labels = SPLICE_JUNCTION_CLASS_MAPPING.keys()
        plt.subplot().legend([box_vanilla['boxes'][0], box_kins['boxes'][0], box_kbann['boxes'][0]], predictors,
                             fontsize=16, loc="lower left")
    else:
        labels = predictors
    plt.subplot().set_xticklabels(labels, fontsize=16)
    plt.savefig(PATH / (name + '-class-accuracy-distributions.pdf'), format='pdf', bbox_inches='tight')
    print("Plot available at " + str(PATH / (name + '-class-accuracy-distributions.pdf')))
