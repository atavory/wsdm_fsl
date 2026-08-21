"""Small deterministic preprocessing helpers for the W16 runner."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_train_test_split(
    indices: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    stratify: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    labels = np.asarray(stratify)
    index_array = np.asarray(indices)
    train_parts = []
    test_parts = []
    for label in np.unique(labels):
        group = index_array[labels == label].copy()
        rng.shuffle(group)
        n_test = int(round(len(group) * test_size))
        if len(group) > 1:
            n_test = min(max(n_test, 1), len(group) - 1)
        else:
            n_test = len(group)
        test_parts.append(group[:n_test])
        train_parts.append(group[n_test:])
    train = np.concatenate(train_parts) if train_parts else np.array([], dtype=index_array.dtype)
    test = np.concatenate(test_parts) if test_parts else np.array([], dtype=index_array.dtype)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_sample_indices(
    indices: np.ndarray,
    *,
    train_size: int,
    random_state: int,
    stratify: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    labels = np.asarray(stratify)
    index_array = np.asarray(indices)
    selected_parts = []
    rest_parts = []
    total = len(index_array)
    remaining = train_size
    unique_labels = np.unique(labels)
    for label_num, label in enumerate(unique_labels):
        group = index_array[labels == label].copy()
        rng.shuffle(group)
        if label_num == len(unique_labels) - 1:
            n_take = remaining
        else:
            n_take = int(round(train_size * len(group) / total))
            n_take = min(max(n_take, 1), len(group))
        n_take = min(n_take, len(group), remaining)
        remaining -= n_take
        selected_parts.append(group[:n_take])
        rest_parts.append(group[n_take:])
    selected = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=index_array.dtype)
    rest = np.concatenate(rest_parts) if rest_parts else np.array([], dtype=index_array.dtype)
    rng.shuffle(selected)
    rng.shuffle(rest)
    return selected, rest


class TabularPreprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.medians: pd.Series | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame, train: np.ndarray) -> None:
        numeric = frame[self.numeric].apply(pd.to_numeric, errors="coerce")
        train_numeric = numeric.iloc[train]
        self.medians = train_numeric.median().fillna(0)
        filled = train_numeric.fillna(self.medians).fillna(0).to_numpy(dtype=np.float32)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        categorical = frame[self.categorical].fillna("MISSING").astype(str)
        self.categories = {
            column: sorted(categorical.iloc[train][column].unique().tolist())
            for column in self.categorical
        }

    def transform(self, frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        if self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform")
        numeric = frame[self.numeric].apply(pd.to_numeric, errors="coerce")
        numeric_part = numeric.iloc[rows].fillna(self.medians).fillna(0).to_numpy(dtype=np.float32)
        numeric_part = (numeric_part - self.means) / self.stds
        categorical = frame[self.categorical].fillna("MISSING").astype(str).iloc[rows]
        one_hot_parts = []
        for column in self.categorical:
            values = categorical[column].to_numpy()
            cats = self.categories[column]
            encoded = np.zeros((len(rows), len(cats)), dtype=np.float32)
            positions = {value: i for i, value in enumerate(cats)}
            for row_idx, value in enumerate(values):
                pos = positions.get(value)
                if pos is not None:
                    encoded[row_idx, pos] = 1.0
            one_hot_parts.append(encoded)
        if one_hot_parts:
            return np.concatenate([numeric_part, *one_hot_parts], axis=1).astype(np.float32)
        return numeric_part.astype(np.float32)


def transform_split(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = TabularPreprocessor(numeric, categorical)
    preprocessor.fit(frame, train)
    return (
        preprocessor.transform(frame, train),
        preprocessor.transform(frame, val),
        preprocessor.transform(frame, test),
    )
