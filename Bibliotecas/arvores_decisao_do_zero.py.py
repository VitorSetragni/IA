from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

@dataclass
class Node:
    is_leaf: bool
    prediction: Any = None
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    split_type: Optional[str] = None
    categories_left: Optional[set] = None
    children: Optional[Dict[Any, "Node"]] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    n_samples: int = 0
    class_counts: Optional[Dict[Any, int]] = None

class DecisionTreeBase:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, min_gain: float = 0.0, feature_types: Optional[List[str]] = None, feature_names: Optional[List[str]] = None, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_gain = float(min_gain)
        self.feature_types = feature_types
        self.feature_names = feature_names
        self.random_state = random_state
        self.root: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_index: Optional[Dict[Any, int]] = None
        self._do_impute: bool = False
        self.impute_values: Dict[int, Any] = {}

    def fit(self, X: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, List[Any]], feature_names: Optional[List[str]] = None, feature_types: Optional[List[str]] = None, impute_missing: bool = True):
        X_arr, names, types = self._prepare_X(X, feature_names, feature_types)
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        self.classes_ = classes
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        y_enc = np.array([self.class_to_index[v] for v in y_arr], dtype=int)
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else None
        idx = np.arange(X_arr.shape[0])
        self.feature_names = names
        self.feature_types = types
        if impute_missing:
            X_arr = X_arr.copy()
            self.impute_values = {}
            for j, t in enumerate(self.feature_types):
                if t == "continuous":
                    col = X_arr[:, j].astype(float)
                    mask = np.isnan(col)
                    if mask.any():
                        fill = float(np.nanmean(col)) if (~mask).any() else 0.0
                        col[mask] = fill
                        X_arr[:, j] = col
                        self.impute_values[j] = fill
                else:
                    col = X_arr[:, j]
                    mask = np.array([(v is None) or (isinstance(v, float) and math.isnan(v)) for v in col], dtype=bool)
                    if mask.any():
                        if (~mask).any():
                            vals = col[~mask]
                            uniques, counts = np.unique(vals, return_counts=True)
                            fill = uniques[np.argmax(counts)]
                        else:
                            fill = "__MISSING__"
                        col2 = col.copy()
                        col2[mask] = fill
                        X_arr[:, j] = col2
                        self.impute_values[j] = fill
            self._do_impute = True
        else:
            self.impute_values = {}
            self._do_impute = False
        used_categorical = set()
        self.root = self._build_node(X_arr, y_enc, idx, depth=0, used_categorical=used_categorical, rng=rng)
        return self

    def predict(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        X_arr = self._to_array(X).copy()
        if getattr(self, "_do_impute", False) and hasattr(self, "impute_values") and self.impute_values:
            for j, fill in self.impute_values.items():
                if self.feature_types[j] == "continuous":
                    col = X_arr[:, j].astype(float)
                    mask = np.isnan(col)
                    if mask.any():
                        col[mask] = float(fill)
                        X_arr[:, j] = col
                else:
                    col = X_arr[:, j]
                    mask = np.array([(v is None) or (isinstance(v, float) and math.isnan(v)) for v in col], dtype=bool)
                    if mask.any():
                        col = col.copy()
                        col[mask] = fill
                        X_arr[:, j] = col
        out = []
        for i in range(X_arr.shape[0]):
            out.append(self._predict_row(X_arr[i], self.root))
        return np.array(out)

    def _predict_row(self, x: np.ndarray, node: Node) -> Any:
        while not node.is_leaf:
            j = node.feature_index
            if self.feature_types[j] == "continuous":
                v = x[j]
                if np.isnan(v):
                    node = node.left if (node.left and (node.left.n_samples >= (node.right.n_samples if node.right else 0))) else (node.right if node.right else node)
                else:
                    node = node.left if v <= node.threshold else node.right
            else:
                v = x[j]
                if node.split_type == "categorical_multi":
                    if v in node.children:
                        node = node.children[v]
                    else:
                        counts = [(child.n_samples, child) for child in node.children.values()]
                        node = max(counts, key=lambda t: t[0])[1]
                else:
                    if v in node.categories_left:
                        node = node.left
                    else:
                        node = node.right
        return self.classes_[np.argmax([node.class_counts.get(c, 0) for c in self.classes_])]

    def _build_node(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int, used_categorical: set, rng: Optional[np.random.RandomState]) -> Node:
        counts = np.bincount(y[indices], minlength=len(self.classes_))
        prediction = self.classes_[np.argmax(counts)]
        if len(np.unique(y[indices])) == 1:
            return Node(True, prediction=prediction, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(True, prediction=prediction, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        if len(indices) < self.min_samples_split:
            return Node(True, prediction=prediction, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        best = self._find_best_split(X, y, indices, used_categorical)
        if best is None or best["gain"] <= self.min_gain:
            return Node(True, prediction=prediction, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        j = best["feature"]
        name = self.feature_names[j]
        if best["type"] == "continuous_binary":
            left_idx = best["left_indices"]
            right_idx = best["right_indices"]
            left = self._build_node(X, y, left_idx, depth + 1, used_categorical, rng)
            right = self._build_node(X, y, right_idx, depth + 1, used_categorical, rng)
            return Node(False, feature_index=j, feature_name=name, threshold=best["threshold"], split_type="continuous_binary", left=left, right=right, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        if best["type"] == "categorical_multi":
            ch = {}
            new_used = set(list(used_categorical) + [j])
            for val, idxs in best["partitions"].items():
                ch[val] = self._build_node(X, y, idxs, depth + 1, new_used, rng)
            return Node(False, feature_index=j, feature_name=name, split_type="categorical_multi", children=ch, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        if best["type"] == "categorical_binary":
            left_idx = best["left_indices"]
            right_idx = best["right_indices"]
            left = self._build_node(X, y, left_idx, depth + 1, used_categorical.union({j}), rng)
            right = self._build_node(X, y, right_idx, depth + 1, used_categorical.union({j}), rng)
            return Node(False, feature_index=j, feature_name=name, split_type="categorical_binary", categories_left=best["categories_left"], left=left, right=right, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})
        return Node(True, prediction=prediction, n_samples=len(indices), class_counts={self.classes_[i]: int(c) for i, c in enumerate(counts)})

    def _entropy(self, counts: np.ndarray) -> float:
        s = counts.sum()
        if s == 0:
            return 0.0
        p = counts[counts > 0] / s
        return float(-np.sum(p * np.log2(p)))

    def _gini(self, counts: np.ndarray) -> float:
        s = counts.sum()
        if s == 0:
            return 0.0
        p = counts / s
        return float(1.0 - np.sum(p * p))

    def _prepare_X(self, X: Union[np.ndarray, "pd.DataFrame"], feature_names: Optional[List[str]], feature_types: Optional[List[str]]):
        try:
            import pandas as pd
        except Exception:
            pd = None
        if pd is not None and isinstance(X, pd.DataFrame):
            names = list(X.columns) if feature_names is None else feature_names
            if feature_types is None:
                types = ["categorical" if str(t).startswith("object") or str(t).startswith("category") else "continuous" for t in X.dtypes]
            else:
                types = list(feature_types)
            arr = X.to_numpy()
            return arr, names, types
        if isinstance(X, np.ndarray):
            arr = X
            n = X.shape[1]
            names = feature_names if feature_names is not None else [f"x{i}" for i in range(n)]
            types = feature_types if feature_types is not None else ["continuous"] * n
            return arr, names, types
        raise TypeError("X must be a numpy array or pandas DataFrame")

    def _to_array(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        try:
            import pandas as pd
        except Exception:
            pd = None
        if pd is not None and isinstance(X, pd.DataFrame):
            return X.to_numpy()
        if isinstance(X, np.ndarray):
            return X
        raise TypeError("X must be a numpy array or pandas DataFrame")

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, used_categorical: set):
        raise NotImplementedError

class ID3Classifier(DecisionTreeBase):
    def fit(self, X: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, List[Any]], feature_names: Optional[List[str]] = None, feature_types: Optional[List[str]] = None):
        return super().fit(X, y, feature_names=feature_names, feature_types=feature_types, impute_missing=False)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, used_categorical: set):
        parent_counts = np.bincount(y[indices], minlength=len(self.classes_))
        parent_entropy = self._entropy(parent_counts)
        best_gain = -1.0
        best = None
        for j, t in enumerate(self.feature_types):
            if t != "categorical" or j in used_categorical:
                continue
            vals = X[indices, j]
            unique_vals = np.unique(vals)
            parts: Dict[Any, np.ndarray] = {}
            child_entropy = 0.0
            for v in unique_vals:
                idxs = indices[vals == v]
                parts[v] = idxs
                counts = np.bincount(y[idxs], minlength=len(self.classes_))
                child_entropy += (len(idxs) / len(indices)) * self._entropy(counts)
            gain = parent_entropy - child_entropy
            if gain > best_gain:
                best_gain = gain
                best = {"feature": j, "type": "categorical_multi", "partitions": parts, "gain": gain}
        return best

class C45Classifier(DecisionTreeBase):
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, used_categorical: set):
        parent_counts = np.bincount(y[indices], minlength=len(self.classes_))
        parent_entropy = self._entropy(parent_counts)
        best_ratio = -1.0
        best = None
        for j, t in enumerate(self.feature_types):
            if t == "categorical":
                if j in used_categorical:
                    continue
                vals = X[indices, j]
                unique_vals = np.unique(vals)
                parts: Dict[Any, np.ndarray] = {}
                child_entropy = 0.0
                split_info = 0.0
                for v in unique_vals:
                    idxs = indices[vals == v]
                    parts[v] = idxs
                    w = len(idxs) / len(indices)
                    counts = np.bincount(y[idxs], minlength=len(self.classes_))
                    child_entropy += w * self._entropy(counts)
                    if w > 0:
                        split_info += -w * math.log2(w)
                gain = parent_entropy - child_entropy
                ratio = gain / split_info if split_info > 0 else 0.0
                if ratio > best_ratio and gain > self.min_gain:
                    best_ratio = ratio
                    best = {"feature": j, "type": "categorical_multi", "partitions": parts, "gain": gain, "ratio": ratio}
            else:
                vals = X[indices, j].astype(float)
                order = np.argsort(vals)
                v_sorted = vals[order]
                y_sorted = y[indices][order]
                total = np.bincount(y_sorted, minlength=len(self.classes_))
                left_counts = np.zeros_like(total)
                best_local_ratio = -1.0
                best_local = None
                for i in range(len(v_sorted) - 1):
                    c = y_sorted[i]
                    left_counts[c] += 1
                    if v_sorted[i] == v_sorted[i + 1]:
                        continue
                    n_left = i + 1
                    n_right = len(v_sorted) - n_left
                    right_counts = total - left_counts
                    ent_left = self._entropy(left_counts)
                    ent_right = self._entropy(right_counts)
                    w_left = n_left / len(v_sorted)
                    w_right = n_right / len(v_sorted)
                    gain = parent_entropy - (w_left * ent_left + w_right * ent_right)
                    split_info = 0.0
                    if w_left > 0:
                        split_info += -w_left * math.log2(w_left)
                    if w_right > 0:
                        split_info += -w_right * math.log2(w_right)
                    ratio = gain / split_info if split_info > 0 else 0.0
                    if ratio > best_local_ratio and gain > self.min_gain:
                        best_local_ratio = ratio
                        thr = (v_sorted[i] + v_sorted[i + 1]) / 2.0
                        left_idx = indices[order[: n_left]]
                        right_idx = indices[order[n_left:]]
                        best_local = {"feature": j, "type": "continuous_binary", "threshold": float(thr), "left_indices": left_idx, "right_indices": right_idx, "gain": gain, "ratio": ratio}
                if best_local is not None and best_local["ratio"] > best_ratio:
                    best_ratio = best_local["ratio"]
                    best = best_local
        return best

class CARTClassifier(DecisionTreeBase):
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, min_gain: float = 0.0, feature_types: Optional[List[str]] = None, feature_names: Optional[List[str]] = None, random_state: Optional[int] = None, max_categorical_enumeration: int = 10):
        super().__init__(max_depth, min_samples_split, min_gain, feature_types, feature_names, random_state)
        self.max_categorical_enumeration = max_categorical_enumeration

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, used_categorical: set):
        parent_counts = np.bincount(y[indices], minlength=len(self.classes_))
        parent_gini = self._gini(parent_counts)
        best_gain = -1.0
        best = None
        for j, t in enumerate(self.feature_types):
            if t == "continuous":
                vals = X[indices, j].astype(float)
                order = np.argsort(vals)
                v_sorted = vals[order]
                y_sorted = y[indices][order]
                total = np.bincount(y_sorted, minlength=len(self.classes_))
                left_counts = np.zeros_like(total)
                for i in range(len(v_sorted) - 1):
                    c = y_sorted[i]
                    left_counts[c] += 1
                    if v_sorted[i] == v_sorted[i + 1]:
                        continue
                    n_left = i + 1
                    n_right = len(v_sorted) - n_left
                    right_counts = total - left_counts
                    g_left = self._gini(left_counts)
                    g_right = self._gini(right_counts)
                    w_left = n_left / len(v_sorted)
                    w_right = n_right / len(v_sorted)
                    impurity = w_left * g_left + w_right * g_right
                    gain = parent_gini - impurity
                    if gain > best_gain and gain > self.min_gain:
                        thr = (v_sorted[i] + v_sorted[i + 1]) / 2.0
                        left_idx = indices[order[: n_left]]
                        right_idx = indices[order[n_left:]]
                        best_gain = gain
                        best = {"feature": j, "type": "continuous_binary", "threshold": float(thr), "left_indices": left_idx, "right_indices": right_idx, "gain": gain}
            else:
                vals = X[indices, j]
                cats = np.unique(vals)
                if len(cats) <= 1:
                    continue
                if len(cats) <= self.max_categorical_enumeration:
                    masks = {c: (vals == c) for c in cats}
                    n = len(vals)
                    total = np.bincount(y[indices], minlength=len(self.classes_))
                    for mask_bits in range(1, 2 ** (len(cats) - 1)):
                        left_mask = np.zeros(n, dtype=bool)
                        left_cats = set()
                        for k, c in enumerate(cats[:-1]):
                            if (mask_bits >> k) & 1:
                                left_mask |= masks[c]
                                left_cats.add(c)
                        right_mask = ~left_mask
                        left_idx = indices[left_mask]
                        right_idx = indices[right_mask]
                        if len(left_idx) == 0 or len(right_idx) == 0:
                            continue
                        left_counts = np.bincount(y[left_idx], minlength=len(self.classes_))
                        right_counts = np.bincount(y[right_idx], minlength=len(self.classes_))
                        g_left = self._gini(left_counts)
                        g_right = self._gini(right_counts)
                        w_left = len(left_idx) / len(indices)
                        w_right = len(right_idx) / len(indices)
                        impurity = w_left * g_left + w_right * g_right
                        gain = parent_gini - impurity
                        if gain > best_gain and gain > self.min_gain:
                            best_gain = gain
                            best = {"feature": j, "type": "categorical_binary", "categories_left": left_cats, "left_indices": left_idx, "right_indices": right_idx, "gain": gain}
                else:
                    totals: Dict[Any, np.ndarray] = {}
                    for c in cats:
                        idxs = indices[vals == c]
                        totals[c] = np.bincount(y[idxs], minlength=len(self.classes_)).astype(float)
                    overall_major = np.argmax(np.bincount(y[indices], minlength=len(self.classes_)))
                    scores = {c: (totals[c][overall_major] / totals[c].sum()) if totals[c].sum() > 0 else 0.0 for c in cats}
                    order = sorted(cats, key=lambda c: scores[c])
                    n = len(order)
                    masks_order = [vals == c for c in order]
                    cum_mask = np.zeros(len(vals), dtype=bool)
                    for i in range(n - 1):
                        cum_mask |= masks_order[i]
                        left_idx = indices[cum_mask]
                        right_idx = indices[~cum_mask]
                        if len(left_idx) == 0 or len(right_idx) == 0:
                            continue
                        left_counts = np.bincount(y[left_idx], minlength=len(self.classes_))
                        right_counts = np.bincount(y[right_idx], minlength=len(self.classes_))
                        g_left = self._gini(left_counts)
                        g_right = self._gini(right_counts)
                        w_left = len(left_idx) / len(indices)
                        w_right = len(right_idx) / len(indices)
                        impurity = w_left * g_left + w_right * g_right
                        gain = parent_gini - impurity
                        if gain > best_gain and gain > self.min_gain:
                            best_gain = gain
                            best = {"feature": j, "type": "categorical_binary", "categories_left": set(order[: i + 1]), "left_indices": left_idx, "right_indices": right_idx, "gain": gain}
        return best

