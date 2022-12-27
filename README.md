# JLC paper experiments

## Description

Three types of connectionist predictors are used to solve two different classification task.
They are: a 3-layers fully connected neural network, a neural network obtained by applying KINS algorithm on the previous predictor and a neural network obtained by KBANN.
Moreover, 3 not connectionist predictors are used: CART and K-nearest neighbours with K equals 3 and 5.
The two classification tasks come from the following datasets:

- [Primate Splice-Junction Gene Sequence](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)) (PSJGS):
a sequence of 60 DNA bases (`adenine`, `cytosine`, `guanine`, `thymine`) is classified into 3 different classes (`exon-intron`, `intron-exon`, `none`).
This dataset comes with a set of classification rules;
- [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) (WBC)
a sequence of (9) features extracted from medical images is classified into two classes (`benign`, `malignant`).
This dataset does not come with rules.

## How to reproduce

1. Install required packages listed in `requirements.txt`.
We suggest to create a new python environment with python 3.9 and then run ```pip install -r requirements.txt```;
2. Generate classification rules for WBC
`python -m setup extract_knowledge`;
3. Run experiments on a dataset `python -m setup run_experiments [-d] [-p] [-s]`,
d specifies the dataset and it is `s` (default is PSJGS) or `b` (WBS), p specifies the population size (default is 30), s specifies the seed for reproducibility (default is 0).
Many files will be created in `results` folder, they can be ignored for now since with the following command they will be summarised into a single file per predictor and per dataset;
4. Generate file with statistics `python -m setup compute_statistics`.
This will create files in `statistics` folder with the semantic of `s_` for PSJGS and `b_` for WBC, followed by the name of the predictor.
Each row is an experiment (with 10-fold cross validation) and in total (if `p` was 30) there are 30 rows plus an additional one that represents the average values of all 30 runs;
5. Generate plots `python -m setup generate_figures`.
This will create images in `figures` folder representing the statistical information of the experiments.
