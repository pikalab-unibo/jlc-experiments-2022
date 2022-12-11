from typing import Iterable
import pandas as pd

UCI_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
SPLICE_JUNCTION_DATA_URL: str = UCI_URL + "molecular-biology/splice-junction-gene-sequences/splice.data"
BREAST_CANCER_DATA_URL: str = UCI_URL + "breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Splice junction variables
SPLICE_JUNCTION_FEATURES = ['a', 'c', 'g', 't']
SPLICE_JUNCTION_CLASS_MAPPING = {'ei': 0, 'ie': 1, 'n': 2}
SPLICE_JUNCTION_AGGREGATE_FEATURE = {'a': ('a',),
                                     'c': ('c',),
                                     'g': ('g',),
                                     't': ('t',),
                                     'd': ('a', 'g', 't'),
                                     'm': ('a', 'c'),
                                     'n': ('a', 'c', 'g', 't'),
                                     'r': ('a', 'g'),
                                     's': ('c', 'g'),
                                     'y': ('c', 't')}
# Breast cancer variables
BREAST_CANCER_CLASS_MAPPING = {'benign': 0, 'malignant': 1}
BREAST_CANCER_CLASS_MAPPING_SHORT = {'b': 0, 'm': 1}


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
                  "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "diagnosis"]
    df.diagnosis = df.diagnosis.apply(lambda x: 0 if x == 2 else 1)
    df.BareNuclei = df.BareNuclei.apply(lambda x: 0 if x == '?' else x).astype(int)
    # One hot encode columns to make it works for KBANN
    new_df = []
    for column in df.columns[:-1]:
        new_df.append(pd.get_dummies(df[column]))
    new_df.append(df.diagnosis)
    new_df = pd.concat(new_df, axis=1)
    new_columns = [i+str(j) for i in df.columns[:-1] for j in list(range(1, 11))]
    new_columns.insert(50, "BareNuclei0")  # For missing values
    new_columns.remove("Mitoses9")  # Always absent
    new_columns.append("diagnosis")
    new_df.columns = new_columns
    return new_df
