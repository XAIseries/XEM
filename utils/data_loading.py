import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def transform_data(dataset, window):
    """
    Transform MTS dataset to tabular dataset

    Parameters
    ----------
    dataset: string
        Name of the dataset

    window: float
        Size of the time window

    Returns
    -------
    sets: array
        Train and test sets
    """

    def transform_mts_features(
        X_mts, X_mts_transformed, line, timestamp, mts_length, window_size, n_features
    ):
        """Add features from previous timestamps"""
        X_mts_transformed[
            line,
            (3 + n_features + (timestamp - 1) * n_features) : (
                3 + 2 * n_features + (timestamp - 1) * n_features
            ),
        ] = X_mts[(line + window_size - 1 - timestamp), 3:]
        return X_mts_transformed

    def transform_mts_line(
        X_mts, X_mts_transformed, line, mts_length, window_size, n_features
    ):
        """Transform MTS per timestamp"""
        return [
            transform_mts_features(
                X_mts,
                X_mts_transformed,
                line,
                timestamp,
                mts_length,
                window_size,
                n_features,
            )
            for timestamp in range(1, window_size)
        ][0]

    def transform_mts(X, mts, mts_length, window_size, n_features):
        """Transform MTS"""
        X_mts = np.array(X[X.id == mts])
        X_mts_transformed = np.empty(
            (mts_length - window_size + 1, 3 + window_size * n_features), dtype=object
        )
        X_mts_transformed[:, : X_mts.shape[1]] = X_mts[window_size - 1 :, :]
        X_mts_transformed = [
            transform_mts_line(
                X_mts, X_mts_transformed, line, mts_length, window_size, n_features
            )
            for line in range(0, X_mts_transformed.shape[0])
        ]
        return X_mts_transformed[0]

    # Load input data
    path = "./data/" + dataset
    df_train = pd.read_parquet(path + "/train.parquet")
    df_test = pd.read_parquet(path + "/test.parquet")

    # Collect input data information
    mts_length = (
        df_train.loc[:, ["id", "timestamp"]]
        .groupby(["id"])
        .count()
        .reset_index(drop=True)
        .max()[0]
    )
    window_size = int(mts_length * window)
    n_features = df_train.iloc[:, 2:-1].shape[1]

    # Transform train and test sets
    train = pd.concat([df_train.target, df_train.iloc[:, :-1]], axis=1)
    test = pd.concat([df_test.target, df_test.iloc[:, :-1]], axis=1)
    train = [
        transform_mts(train, mts, mts_length, window_size, n_features)
        for mts in np.unique(train.id)
    ]
    train = np.concatenate(train, axis=0)
    test = [
        transform_mts(test, mts, mts_length, window_size, n_features)
        for mts in np.unique(test.id)
    ]
    test = np.concatenate(test, axis=0)

    # Separate X and y
    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_test = test[:, 1:]
    y_test = test[:, 0]

    return X_train, y_train, X_test, y_test


def load_data(dataset, window):
    """
    Import train and test sets

    Parameters
    ----------
    dataset: string
        Name of the dataset

    window: float
        Size of the time window

    Returns
    -------
    sets: array
        Train and test sets
    """
    path = "./data/" + dataset + "/transformed/" + str(int(window * 100))

    if not os.path.exists(path):
        # Transform the dataset and save it
        os.makedirs(path)
        X_train, y_train, X_test, y_test = transform_data(dataset, window)
        np.save(path + "/X_train.npy", X_train)
        np.save(path + "/y_train.npy", y_train)
        np.save(path + "/X_test.npy", X_test)
        np.save(path + "/y_test.npy", y_test)
    else:
        # Load existing transformed dataset
        X_train = np.load(path + "/X_train.npy", allow_pickle=True)
        y_train = np.load(path + "/y_train.npy", allow_pickle=True)
        X_test = np.load(path + "/X_test.npy", allow_pickle=True)
        y_test = np.load(path + "/y_test.npy", allow_pickle=True)

    return X_train, y_train, X_test, y_test


def import_data(dataset, window, xp_dir, val_split=[3, 1], log=print):
    """
    Generate train, validation and test sets

    Parameters
    ----------
    dataset: string
        Name of the dataset

    window: float
        Size of the time window

    xp_dir: string
        Folder of the experiment

    val_split: array
        Number of folds and the selected one

    log: string
        Processing of the outputs

    Returns
    -------
    sets: array
        Train, validation and test sets
    """
    # Load train and test sets
    X_train, y_train, X_test, y_test = load_data(dataset, window)

    # Print input data information
    classes, y = np.unique(y_train, return_inverse=True)
    mts_length = (len(X_train) / len(np.unique(X_train[:, 0])) - 1) / (1 - window)
    window_size = window * mts_length
    n_features = (X_train.shape[1] - 2) / window_size
    log("Number of MTS in train set: {0}".format(len(np.unique(X_train[:, 0]))))
    log("Number of MTS in test set: {0}".format(len(np.unique(X_test[:, 0]))))
    log("Number of classes: {0}".format(len(classes)))
    log("MTS length: {0}".format(int(mts_length)))
    log("Window size: {0}".format(int(window_size)))
    log("Number of features: {0}".format(int(n_features)))

    # Generate train/validation split
    df_split = pd.concat([pd.DataFrame(X_train[:, 0]), pd.DataFrame(y)], axis=1)
    df_split.columns = ["id", "target"]
    df_split = df_split.groupby(["id"]).mean().reset_index(drop=False)

    if val_split[1] == 0:
        y_train, y_val = y_train, []
        X_train, X_val = X_train, []
    else:
        skf = StratifiedKFold(n_splits=val_split[0], shuffle=False)
        train_index, val_index = list(skf.split(df_split.id, df_split.target))[
            val_split[1] - 1
        ]
        y_train, y_val = (
            y_train[np.isin(X_train[:, 0], df_split.iloc[train_index, 0].values)],
            y_train[~np.isin(X_train[:, 0], df_split.iloc[train_index, 0].values)],
        )
        X_train, X_val = (
            X_train[np.isin(X_train[:, 0], df_split.iloc[train_index, 0].values)],
            X_train[~np.isin(X_train[:, 0], df_split.iloc[train_index, 0].values)],
        )

    log(
        "\nCross-validation - folds: {0}, current: {1}".format(
            val_split[0], val_split[1]
        )
    )
    log("Training set size: {0}".format(len(X_train)))
    log("Validation set size: {0}".format(len(X_val)))
    log("Testing set size: {0}".format(len(X_test)))

    return X_train, y_train, X_val, y_val, X_test, y_test
