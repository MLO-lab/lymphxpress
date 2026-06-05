import click
import logging

from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import ray
except ImportError:
    ray = None

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


def compute_required_features_per_model(
    total_features: int, n_models: int, target_coverage: float = 0.95
) -> int:
    """
    Compute the number of features that must be sampled per model to achieve
    a desired expected feature coverage across an ensemble.

    Parameters
    ----------
    total_features : int
        Total number of available features (D).
    n_models : int
        Number of ensemble models (M).
    target_coverage : float
        Desired coverage level (default: 0.95).

    Returns
    -------
    int
        Minimum required features per model (rounded up).
    """
    if not (0 < target_coverage < 1):
        raise ValueError("target_coverage must be between 0 and 1.")

    D = float(total_features)
    M = float(n_models)

    k = D * (1 - (1 - target_coverage) ** (1.0 / M))
    return int(np.ceil(k))


RNASEQ_DGE = "rnaseq_dge"
RNAPANEL = "rnapanel"


class ConsensusClassifier:
    def __init__(
        self,
        n_models=100,
        n_samples_per_model=None,
        n_features_per_model="auto",
        keep_samples=None,
        keep_features=None,
        verbose=True,
        xgb_params_override=None,
        use_ray=False,
    ):
        if n_models < 1:
            raise ValueError(f"n_models must be at least 1, got {n_models}.")
        self.n_models = n_models
        self.n_samples_per_model = n_samples_per_model
        self.n_features_per_model = n_features_per_model
        self.keep_samples = keep_samples
        self.keep_features = keep_features

        self.verbose = verbose
        self.log = logger.info if self.verbose else lambda *a, **k: None

        self.log("========== Initializing Consensus Classifier ==========")
        self.log(
            "Finished! Call the `add_train_data` method to load the training data."
        )

        self.xgb_params_override = xgb_params_override
        if self.xgb_params_override is None:
            self.xgb_params_override = {}
        self.use_ray = use_ray
        if self.use_ray and ray is None:
            logger.warning(
                "Ray not available, disabling `use_ray` in ConsensusClassifier."
            )
            self.use_ray = False
        self._reset()

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.le = None
        self.le_mapping = None
        self.n_classes = None
        self.binary = None

        self.X_test = None

        self.sample_subset_list = []
        self.feature_subset_list = []
        self.classifier_list = []

        self.probs_list = []
        self._probs_df = None
        self._pred_series = None
        self._vote_series = None
        self._all_votes_df = None

    @property
    def xgb_config(self):
        if self.binary:
            scale_pos_weight = float(np.sum(self.y_train == 0)) / max(
                float(np.sum(self.y_train == 1)), 1.0
            )
            xgb_config = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "scale_pos_weight": scale_pos_weight,
            }
        else:
            xgb_config = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": self.n_classes,
            }

        defaults = {
            "eta": 0.1,
            "gamma": 1.0,
            # l1 regularization term on weights
            "alpha": 1.0,
            # l2 regularization term on weights
            "lambda": 5.0,
            "max_depth": 4,
            "subsample": 0.8,
            "n_estimators": 200,
            "tree_method": "hist",
        }

        xgb_config.update(defaults)
        if self.xgb_params_override:
            xgb_config.update(self.xgb_params_override)

        return xgb_config

    @property
    def probs(self):
        """Return ensemble-averaged probability estimates as a DataFrame."""
        if self._probs_df is None:
            raise RuntimeError(
                "predict_proba() must be called before accessing `.probs`."
            )
        return self._probs_df

    @property
    def predictions(self):
        """Return ensemble class predictions as a Series of labels."""
        if self._pred_series is None:
            raise RuntimeError(
                "predict() must be called before accessing `.predictions`."
            )
        return self._pred_series

    @property
    def votes(self):
        """Return ensemble vote counts for the winning label per sample."""
        if self._vote_series is None:
            raise RuntimeError("predict() must be called before accessing `.votes`.")
        return self._vote_series

    @property
    def all_votes(self):
        """
        Return per-class vote counts for each sample.
        Columns are class labels.
        """
        if self._all_votes_df is None:
            raise RuntimeError(
                "predict() must be called before accessing `.all_votes`."
            )
        return self._all_votes_df

    def _check_data(self, X):
        invalid_rows = X.index.isna()
        if any(invalid_rows):
            logger.warning(f"Found {invalid_rows.sum()} invalid rows, removing...")
        invalid_columns = X.columns.isna()
        if any(invalid_columns):
            logger.warning(
                f"Found {invalid_columns.sum()} invalid columns, removing..."
            )
        return X.loc[~invalid_rows, ~invalid_columns].copy()

    def _intersect_features(self, X_train, X_test):
        common_features = [col for col in X_test.columns if col in X_train.columns]
        n_common = len(common_features)
        n_train_features = len(X_train.columns)
        percentage = 100 * (n_common / n_train_features)
        if n_common < n_train_features:
            self.log(
                f"Found {n_common} out of {n_train_features} features ({percentage:.2f}%) in the test data."
            )
        if n_common == 0:
            raise ValueError(
                "No common features found between training and test data. "
                "Check that column names match."
            )
        self.log("Subsetting features of the test data...")
        X_train = X_train.loc[:, common_features].copy()
        X_test = X_test.loc[:, common_features].copy()

        if self.keep_features is not None:
            self.keep_features = [
                kf for kf in self.keep_features if kf in common_features
            ]

        if self.n_features_per_model == "auto":
            self.n_features_per_model = compute_required_features_per_model(
                total_features=X_train.shape[1],
                n_models=self.n_models,
                target_coverage=0.95,
            )

        if (
            self.n_features_per_model is None
            or self.n_features_per_model <= 0
            or self.n_features_per_model > X_train.shape[1]
        ):
            self.n_features_per_model = X_train.shape[1]

        if 0 < self.n_features_per_model <= 1.0:
            self.n_features_per_model = int(
                self.n_features_per_model * X_train.shape[1]
            )

        self.log(f"Each model will be trained on {self.n_features_per_model} features.")

        return X_train, X_test

    def add_train_data(self, X, y, y_multi=None):
        self.log("Checking training data...")
        self.X_train = self._check_data(X)
        # align y to the rows that survived _check_data
        self.y_train = y.loc[self.X_train.index].copy()
        self.y_multi = y_multi.loc[self.X_train.index].copy() if y_multi is not None else None
        self.le = LabelEncoder()
        self.le.fit(self.y_train)
        self.y_train = pd.Series(
            self.le.transform(self.y_train), index=self.y_train.index
        )
        self.n_classes = len(self.le.classes_)
        self.binary = self.n_classes == 2
        self.le_mapping = dict(zip(range(len(self.le.classes_)), self.le.classes_))

        if (
            self.n_samples_per_model is None
            or self.n_samples_per_model <= 0
            or self.n_samples_per_model > self.X_train.shape[0]
        ):
            self.n_samples_per_model = self.X_train.shape[0]

        if 0 < self.n_samples_per_model < 1.0:
            self.n_samples_per_model = int(
                self.n_samples_per_model * self.X_train.shape[0]
            )

        self.log(
            f"Loaded training data with {self.X_train.shape[0]} samples and {self.X_train.shape[1]} features."
        )
        self.log(f"Found {self.n_classes} classes in the training data.")
        self.log(f"Each model will be trained on {self.n_samples_per_model} samples.")
        self.log("Finished! Call the `add_test_data` method to load the test data.")

    def add_test_data(self, X):
        if self.X_train is None:
            raise RuntimeError("add_train_data() must be called before add_test_data().")
        self.log("Checking test data...")
        self.X_test = self._check_data(X)
        self.log(
            f"Loaded test data with {self.X_test.shape[0]} samples and {self.X_test.shape[1]} features."
        )

        self.X_train, self.X_test = self._intersect_features(self.X_train, self.X_test)

        self.probs_list = []
        self._probs_df = None
        self._pred_series = None
        self._vote_series = None
        self._all_votes_df = None

        self.log("Finished! Call the `fit` method to train the classifiers.")

    if ray is not None:

        @staticmethod
        @ray.remote
        def _fit_classifier_remote(xgb_config, X, y, model_idx, use_sample_weight):
            clf = xgb.XGBClassifier(**xgb_config)
            if use_sample_weight:
                sample_weight = compute_sample_weight(class_weight="balanced", y=y)
                return clf.fit(X, y, sample_weight=sample_weight), model_idx
            return clf.fit(X, y), model_idx

    @staticmethod
    def _fit_classifier(xgb_config, X, y, use_sample_weight):
        clf = xgb.XGBClassifier(**xgb_config)
        if use_sample_weight:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
            return clf.fit(X, y, sample_weight=sample_weight)
        return clf.fit(X, y)

    def _sample_samples_per_model(self, rng):
        """
        Stratified sampling of samples for one model.
        Uses y_multi when provided, otherwise y_train.
        Ensures at least one sample per class.
        """
        y_eff = self.y_multi if self.y_multi is not None else self.y_train
        percentage = self.n_samples_per_model / len(self.y_train)

        sample_subset = []
        for label in y_eff.unique():
            idx = y_eff.index[y_eff == label].to_numpy()
            n_samples = max(1, int(len(idx) * percentage))
            n_samples = min(n_samples, len(idx))
            chosen = rng.choice(idx, size=n_samples, replace=False)
            sample_subset.extend(chosen.tolist())

        if self.keep_samples is not None:
            sample_subset = list(set(sample_subset) | set(self.keep_samples))

        return sample_subset

    def _sample_features_per_model(self, rng):
        """
        Randomly sample features for one model, optionally union with keep_features.
        """
        cols = self.X_train.columns.to_numpy()
        features = rng.choice(
            cols, size=self.n_features_per_model, replace=False
        ).tolist()

        if self.keep_features is not None:
            features = list(set(features) | set(self.keep_features))

        return features

    def fit(self, seed=None, on_progress=None, n_jobs=1):
        """
        Train the ensemble of XGBoost classifiers by repeatedly sampling
        stratified sample subsets and feature subsets.

        on_progress: optional callable(current: int, total: int) called after each model.
        n_jobs: threads for parallel training (XGBoost releases the GIL).
                0 or 1 = sequential. -1 = all available cores. Ignored when use_ray=True.
        """
        if self.X_train is None:
            raise RuntimeError("add_train_data() must be called before fit().")
        if self.X_test is None:
            raise RuntimeError("add_test_data() must be called before fit().")
        rng = np.random.default_rng(seed)
        self.log("Training the classifiers...")

        self.sample_subset_list = []
        self.feature_subset_list = []
        self.classifier_list = []
        self.probs_list = []
        self._probs_df = None
        self._pred_series = None
        self._vote_series = None
        self._all_votes_df = None

        if on_progress is not None:
            on_progress(0, self.n_models)

        # pre-sample all subsets so ordering is deterministic regardless of path
        for model_idx in range(self.n_models):
            self.sample_subset_list.append(self._sample_samples_per_model(rng))
            self.feature_subset_list.append(self._sample_features_per_model(rng))

        # ---- Ray path ----
        if self.use_ray and ray is not None:
            ray_tasks = []
            for model_idx in range(self.n_models):
                X_sub = self.X_train.loc[self.sample_subset_list[model_idx], self.feature_subset_list[model_idx]]
                y_sub = self.y_train.loc[self.sample_subset_list[model_idx]]
                cfg = self.xgb_config.copy()
                cfg["random_state"] = (seed or 0) + model_idx
                ray_tasks.append(
                    self._fit_classifier_remote.remote(
                        xgb_config=cfg,
                        X=X_sub,
                        y=y_sub,
                        model_idx=model_idx,
                        use_sample_weight=not self.binary,
                    )
                )

            results = [None] * self.n_models
            remaining = list(ray_tasks)
            pbar = tqdm(total=len(remaining), desc="Training ensemble", disable=not self.verbose)
            while remaining:
                done, remaining = ray.wait(remaining, num_returns=1)
                clf, idx = ray.get(done[0])
                results[idx] = clf
                pbar.update(1)
                if on_progress is not None:
                    on_progress(len(ray_tasks) - len(remaining), self.n_models)
            pbar.close()
            self.classifier_list = results
            self.log("Finished! Call the `predict` method to classify the test data.")
            return self.classifier_list

        # ---- Thread-parallel path ----
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        use_parallel = n_jobs > 1

        def _train_one(model_idx):
            X_sub = self.X_train.loc[self.sample_subset_list[model_idx], self.feature_subset_list[model_idx]]
            y_sub = self.y_train.loc[self.sample_subset_list[model_idx]]
            cfg = self.xgb_config.copy()
            cfg["random_state"] = (seed or 0) + model_idx
            # one thread per model; XGBoost's own threading handles intra-model parallelism
            if use_parallel:
                cfg.setdefault("nthread", 1)
            return model_idx, self._fit_classifier(cfg, X_sub, y_sub, use_sample_weight=not self.binary)

        results = [None] * self.n_models
        completed = 0
        lock = Lock()

        if use_parallel:
            pbar = tqdm(total=self.n_models, desc="Training ensemble", disable=not self.verbose)
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(_train_one, i): i for i in range(self.n_models)}
                for future in as_completed(futures):
                    idx, clf = future.result()
                    results[idx] = clf
                    with lock:
                        completed += 1
                        pbar.update(1)
                        if on_progress is not None:
                            on_progress(completed, self.n_models)
            pbar.close()
        else:
            for model_idx in tqdm(range(self.n_models), desc="Training ensemble", disable=not self.verbose):
                _, clf = _train_one(model_idx)
                results[model_idx] = clf
                completed += 1
                if on_progress is not None:
                    on_progress(completed, self.n_models)

        self.classifier_list = results
        self.log("Finished! Call the `predict` method to classify the test data.")
        return self.classifier_list

    def _predict_proba(self):
        """
        Generate per-model probability predictions for the test set.
        Results are stored in `self.probs_list` as numpy arrays.
        """
        X_test_full = self.X_test  # alias

        self.probs_list = []
        append = self.probs_list.append  # local binding is faster

        for features, clf in zip(self.feature_subset_list, self.classifier_list):
            X_sub = X_test_full.loc[:, features]

            append(clf.predict_proba(X_sub))

        return self.probs_list

    @staticmethod
    def _safe_logit(p):
        """Convert probability p to logit with clipping for stability."""
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return np.log(p / (1 - p))

    @staticmethod
    def _safe_sigmoid(x):
        """Stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-x))

    def _aggregate_binary_logits(self):
        """
        Aggregate binary ensemble predictions in logit space.
        probs_list: list of (n_samples, 2) arrays.
        Returns: (n_samples, 2) array.
        """
        # extract P(positive) for each model → shape (n_models, n_samples)
        p1 = np.stack([p[:, 1] for p in self.probs_list], axis=0)

        # convert each to logits
        logits = self._safe_logit(p1)

        # mean logit across models
        mean_logits = logits.mean(axis=0)

        # convert back to probability
        p1_final = self._safe_sigmoid(mean_logits)
        p0_final = 1.0 - p1_final

        return np.stack([p0_final, p1_final], axis=1)

    def _aggregate_multiclass_logits(self):
        """
        Aggregate multiclass probabilities in logit space.
        probs_list: list of (n_samples, n_classes)
        Returns: (n_samples, n_classes)
        """
        probs = np.stack(self.probs_list, axis=0)  # shape (M, N, C)

        # clip
        probs = np.clip(probs, 1e-8, 1 - 1e-8)

        # compute logits per class
        logits = np.log(probs)

        # average logits across models
        mean_logits = logits.mean(axis=0)  # (N, C)

        # normalize back to probabilities
        exp_logits = np.exp(mean_logits)
        final = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return final

    def predict_proba(self):
        if not self.classifier_list:
            raise ValueError("No classifiers trained yet. Call fit() first.")

        if len(self.probs_list) == 0:
            self._predict_proba()

        if self.binary:
            agg_probs = self._aggregate_binary_logits()
        else:
            agg_probs = self._aggregate_multiclass_logits()

        df = pd.DataFrame(
            agg_probs,
            index=self.X_test.index,
            columns=[self.le_mapping[i] for i in range(agg_probs.shape[1])],
        )

        self._probs_df = df
        return df

    def predict(self):
        """
        Compute ensemble hard predictions and vote counts.
        Break vote ties using aggregated ensemble probabilities.
        Populates:
        - _all_votes_df : per-class vote counts
        - _vote_series  : votes for winning class
        - _pred_series  : predicted label
        """
        if not self.classifier_list:
            raise RuntimeError("fit() must be called before predict().")

        _ = self.predict_proba()  # ensure probs_list exists

        # (n_models, n_samples)
        hard_preds = np.stack([np.argmax(p, axis=1) for p in self.probs_list], axis=0)

        # --- per-class vote counts ---
        votes = {}
        for class_idx, class_label in self.le_mapping.items():
            votes[class_label] = (hard_preds == class_idx).sum(axis=0)

        all_votes_df = pd.DataFrame(votes, index=self.X_test.index)

        # --- winner: break vote ties via aggregated probabilities ---
        max_votes = all_votes_df.max(axis=1)
        is_tied = (all_votes_df == max_votes.values[:, None]).sum(axis=1) > 1
        pred_labels = all_votes_df.idxmax(axis=1)
        if is_tied.any():
            prob_winner = self._probs_df.loc[is_tied].idxmax(axis=1)
            pred_labels.loc[is_tied] = prob_winner
        winner_votes = all_votes_df.max(axis=1)

        self._all_votes_df = all_votes_df
        self._pred_series = pred_labels
        self._vote_series = winner_votes

        return self._pred_series

    def export_results(self, path=None):
        preds = self.predict()
        probs = self.probs
        votes = self.all_votes

        results = pd.concat(
            [preds.rename("Prediction"), votes.add_prefix("votes_"), probs],
            axis=1,
        )

        if path is not None:
            results.to_csv(path)

        return results


def scale(_X, scaler):
    index = _X.index
    columns = _X.columns
    return pd.DataFrame(scaler.transform(_X), index=index, columns=columns)


def intersect_features(current_features, new_features):
    current_set = set(current_features)
    return [col for col in new_features if col in current_set]


def _subset_scaler(scaler, full_features, subset_features):
    """Return a new StandardScaler whose statistics are sliced to subset_features."""
    idx_map = {f: i for i, f in enumerate(full_features)}
    indices = np.array([idx_map[f] for f in subset_features])
    sub = StandardScaler()
    sub.mean_ = scaler.mean_[indices]
    sub.scale_ = scaler.scale_[indices]
    sub.var_ = scaler.var_[indices]
    sub.n_features_in_ = len(indices)
    sub.n_samples_seen_ = scaler.n_samples_seen_
    if hasattr(scaler, "feature_names_in_"):
        sub.feature_names_in_ = scaler.feature_names_in_[indices]
    return sub


class LymphxpressClassifier:
    _CACHE: dict = {}  # process-level cache: keyed by "default" for the standard 4-CSV case

    def __init__(
        self,
        n_models=100,
        n_features_per_model="auto",
        verbose=True,
        xgb_params_override_n1=None,
        xgb_params_override_multi=None,
        use_ray=False,
        keep_train_classes=None,
        drop_train_classes=None,
        training_X=None,
        training_y=None,
        training_key="custom_rnaseq",
    ):
        self.n_models = n_models
        self.n_features_per_model = n_features_per_model
        self.verbose = verbose
        self.log = logger.info if self.verbose else lambda *a, **k: None
        self.xgb_params_override_n1 = xgb_params_override_n1
        self.xgb_params_override_multi = xgb_params_override_multi
        self.use_ray = use_ray
        self.keep_train_classes = (
            set(keep_train_classes) if keep_train_classes is not None else None
        )
        self.drop_train_classes = (
            set(drop_train_classes) if drop_train_classes is not None else set()
        )
        self.custom_training = training_X is not None or training_y is not None
        if self.custom_training and (training_X is None or training_y is None):
            raise ValueError("training_X and training_y must be provided together.")

        if self.custom_training:
            self._cacheable = False
            training_X = pd.DataFrame(training_X).copy()
            training_X = training_X.loc[
                ~training_X.index.isna(),
                ~training_X.columns.isna(),
            ].copy()
            training_y = pd.Series(training_y).reindex(training_X.index).astype(str)
            if training_y.isna().any():
                raise ValueError("training_y contains missing labels after alignment.")
            self.data_keys = [training_key]
            self.X_train_dict_raw = {training_key: training_X.copy()}
            self.y_train_dict = {training_key: training_y.copy()}
        else:
            self.data_keys = ["cornell_pseudobulk", "vancouver", "cornell_bulk", "ffm"]
            # Cache is valid only for the default case (no class filtering)
            _cache_ok = self.keep_train_classes is None and not self.drop_train_classes
            self._cacheable = _cache_ok
            _c = LymphxpressClassifier._CACHE.get("default") if _cache_ok else None
            if _c is not None:
                self.X_train_dict_raw = _c["X_train_dict_raw"]
                self.y_train_dict = _c["y_train_dict"]
            else:
                self.X_train_dict_raw = {
                    k: pd.read_csv(_DATA_DIR / f"X_{k}.csv", index_col=0) for k in self.data_keys
                }
                self.y_train_dict = {
                    k: pd.read_csv(_DATA_DIR / f"y_{k}.csv", index_col=0).iloc[:, 0]
                    for k in self.data_keys
                }
        self._filter_training_labels()

        if self._cacheable and "common_features" in (LymphxpressClassifier._CACHE.get("default") or {}):
            self.common_features = list(LymphxpressClassifier._CACHE["default"]["common_features"])
        else:
            first_key = self.data_keys[0]
            self.common_features = self.X_train_dict_raw[first_key].columns
            for k, df in self.X_train_dict_raw.items():
                if k != first_key:
                    self.common_features = intersect_features(self.common_features, df.columns)
            if self._cacheable:
                LymphxpressClassifier._CACHE.setdefault("default", {}).update({
                    "X_train_dict_raw": self.X_train_dict_raw,
                    "y_train_dict": self.y_train_dict,
                    "common_features": list(self.common_features),
                })

        # save the full training-data feature set before per-job test intersection
        self._full_common_features = list(self.common_features)

        # placeholders for later
        self.scalers = {}
        self.X_train = None
        self.y_train = None
        self.n1_classifier = None
        self.multi_classifier = None
        self._test_data_added = False

    def _filter_training_labels(self):
        if self.keep_train_classes is None and not self.drop_train_classes:
            return

        for key in self.data_keys:
            y = self.y_train_dict[key].copy()
            keep_mask = pd.Series(True, index=y.index)
            if self.keep_train_classes is not None:
                keep_mask &= y.isin(self.keep_train_classes)
            if self.drop_train_classes:
                keep_mask &= ~y.isin(self.drop_train_classes)

            self.X_train_dict_raw[key] = self.X_train_dict_raw[key].loc[keep_mask].copy()
            self.y_train_dict[key] = y.loc[keep_mask].copy()

    def _setup_training_data(self):
        if self.custom_training:
            X_dict = {
                k: self.X_train_dict_raw[k].loc[:, self.common_features].copy()
                for k in self.data_keys
            }
            X_all = pd.concat([X_dict[k] for k in self.data_keys])
            scaler = StandardScaler().fit(X_all)
            X_dict = {k: scale(v, scaler) for k, v in X_dict.items()}
            self.X_train = pd.concat([X_dict[k] for k in self.data_keys])
            self.y_train = pd.concat([self.y_train_dict[k] for k in self.data_keys])
            self.scalers = {RNASEQ_DGE: scaler, RNAPANEL: scaler}
            return

        if not self._cacheable:
            # Class-filtering path: compute fresh on the already-intersected feature set
            X_dict = {
                k: self.X_train_dict_raw[k].loc[:, self.common_features].copy()
                for k in self.data_keys
            }
            scaler_dict = {k: StandardScaler().fit(v) for k, v in X_dict.items()}
            scaler_dict[RNASEQ_DGE] = StandardScaler().fit(
                pd.concat([X_dict["vancouver"], X_dict["cornell_bulk"]])
            )
            scaler_dict[RNAPANEL] = scaler_dict["ffm"]
            X_dict["cornell_pseudobulk"] = scale(X_dict["cornell_pseudobulk"], scaler_dict["cornell_pseudobulk"])
            X_dict["vancouver"] = scale(X_dict["vancouver"], scaler_dict[RNASEQ_DGE])
            X_dict["cornell_bulk"] = scale(X_dict["cornell_bulk"], scaler_dict[RNASEQ_DGE])
            X_dict["ffm"] = scale(X_dict["ffm"], scaler_dict[RNAPANEL])
            self.X_train = pd.concat([X_dict[k] for k in self.data_keys])
            self.y_train = pd.concat([self.y_train_dict[k] for k in self.data_keys])
            self.scalers = scaler_dict
            return

        # Cacheable standard path
        full_features = self._full_common_features
        _c = LymphxpressClassifier._CACHE.get("default", {})

        if "X_train_full" in _c:
            # Fast path: subset from cached full-feature data
            self.X_train = _c["X_train_full"].loc[:, self.common_features].copy()
            self.y_train = _c["y_train_full"].copy()
            self.scalers = {
                k: _subset_scaler(v, full_features, self.common_features)
                for k, v in _c["scaler_dict"].items()
            }
            return

        # Slow path (first job, cache miss): compute on full features, cache, then subset
        X_dict = {k: self.X_train_dict_raw[k].loc[:, full_features].copy() for k in self.data_keys}
        scaler_dict = {k: StandardScaler().fit(v) for k, v in X_dict.items()}
        scaler_dict[RNASEQ_DGE] = StandardScaler().fit(
            pd.concat([X_dict["vancouver"], X_dict["cornell_bulk"]])
        )
        scaler_dict[RNAPANEL] = scaler_dict["ffm"]
        X_dict["cornell_pseudobulk"] = scale(X_dict["cornell_pseudobulk"], scaler_dict["cornell_pseudobulk"])
        X_dict["vancouver"] = scale(X_dict["vancouver"], scaler_dict[RNASEQ_DGE])
        X_dict["cornell_bulk"] = scale(X_dict["cornell_bulk"], scaler_dict[RNASEQ_DGE])
        X_dict["ffm"] = scale(X_dict["ffm"], scaler_dict[RNAPANEL])
        X_train_full = pd.concat([X_dict[k] for k in self.data_keys])
        y_train_full = pd.concat([self.y_train_dict[k] for k in self.data_keys])

        LymphxpressClassifier._CACHE.setdefault("default", {}).update({
            "X_train_full": X_train_full,
            "y_train_full": y_train_full,
            "scaler_dict": dict(scaler_dict),
        })

        self.X_train = X_train_full.loc[:, self.common_features].copy()
        self.y_train = y_train_full.copy()
        self.scalers = {
            k: _subset_scaler(v, full_features, self.common_features)
            for k, v in scaler_dict.items()
        }

    @classmethod
    def _prewarm(cls):
        """Pre-populate the process-level cache: reads CSVs, fits scalers, scales training data.
        Call this at worker startup so the first job pays no setup cost."""
        if "X_train_full" in cls._CACHE.get("default", {}):
            return
        data_keys = ["cornell_pseudobulk", "vancouver", "cornell_bulk", "ffm"]
        X_train_dict_raw = {k: pd.read_csv(_DATA_DIR / f"X_{k}.csv", index_col=0) for k in data_keys}
        y_train_dict = {
            k: pd.read_csv(_DATA_DIR / f"y_{k}.csv", index_col=0).iloc[:, 0]
            for k in data_keys
        }
        first_key = data_keys[0]
        common_features = list(X_train_dict_raw[first_key].columns)
        for k, df in X_train_dict_raw.items():
            if k != first_key:
                common_features = intersect_features(common_features, df.columns)
        X_dict = {k: X_train_dict_raw[k].loc[:, common_features].copy() for k in data_keys}
        scaler_dict = {k: StandardScaler().fit(v) for k, v in X_dict.items()}
        scaler_dict[RNASEQ_DGE] = StandardScaler().fit(
            pd.concat([X_dict["vancouver"], X_dict["cornell_bulk"]])
        )
        scaler_dict[RNAPANEL] = scaler_dict["ffm"]
        X_dict["cornell_pseudobulk"] = scale(X_dict["cornell_pseudobulk"], scaler_dict["cornell_pseudobulk"])
        X_dict["vancouver"] = scale(X_dict["vancouver"], scaler_dict[RNASEQ_DGE])
        X_dict["cornell_bulk"] = scale(X_dict["cornell_bulk"], scaler_dict[RNASEQ_DGE])
        X_dict["ffm"] = scale(X_dict["ffm"], scaler_dict[RNAPANEL])
        X_train_full = pd.concat([X_dict[k] for k in data_keys])
        y_train_full = pd.concat([y_train_dict[k] for k in data_keys])
        cls._CACHE["default"] = {
            "X_train_dict_raw": X_train_dict_raw,
            "y_train_dict": y_train_dict,
            "common_features": common_features,
            "X_train_full": X_train_full,
            "y_train_full": y_train_full,
            "scaler_dict": dict(scaler_dict),
        }

    def _init_n1_classifier(self):
        n1_signature = pd.read_csv(
            _DATA_DIR / "20250122_Wright_GeneWeights.csv", index_col=0
        )
        keep_features = n1_signature.loc[
            n1_signature["Type"] == "N1", "genename.sing"
        ].tolist()
        override = dict(self.xgb_params_override_n1) if self.xgb_params_override_n1 else {}
        n_samples_per_model = override.pop("n_samples_per_model", 0.4)
        if len(keep_features) == 0:
            keep_features = None
        clf = ConsensusClassifier(
            n_models=self.n_models,
            n_samples_per_model=n_samples_per_model,
            n_features_per_model=self.n_features_per_model,
            keep_samples=self.y_train[self.y_train == "N1"].index.tolist(),
            keep_features=keep_features,
            verbose=self.verbose,
            xgb_params_override=override,
            use_ray=self.use_ray,
        )
        self.log("Initialized the N1 classifier!")
        return clf

    def _init_multi_classifier(self):
        clf = ConsensusClassifier(
            n_models=self.n_models,
            n_features_per_model=self.n_features_per_model,
            verbose=self.verbose,
            xgb_params_override=self.xgb_params_override_multi,
            use_ray=self.use_ray,
        )
        self.log("Initialized the multi-class classifier!")
        return clf

    def _setup_classifiers(self):
        if self.n1_classifier is None:
            self.n1_classifier = self._init_n1_classifier()
        if self.multi_classifier is None:
            self.multi_classifier = self._init_multi_classifier()
        self.log("Classifiers are set up and ready for training.")

    def add_train_data(self):
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        self.n1_classifier.add_train_data(
            X_train, (y_train == "N1").copy(), y_multi=y_train
        )
        non_N1 = ~(self.y_train == "N1")
        X_train = self.X_train.loc[non_N1, :].copy()
        y_train = self.y_train.loc[non_N1].copy()
        self.multi_classifier.add_train_data(X_train, y_train, y_multi=y_train)
        self.log("Training data added.")

    def add_test_data(self, X_test, scaler=None):
        if self._test_data_added:
            raise RuntimeError(
                "add_test_data() has already been called on this LymphxpressClassifier instance. "
                "Please create a new LymphxpressClassifier for a new test dataset."
            )
        self.common_features = intersect_features(self.common_features, X_test.columns)
        if len(self.common_features) == 0:
            raise ValueError("No common features found between training and test data.")
        self._setup_training_data()
        self._setup_classifiers()
        self.add_train_data()
        X_test = X_test.loc[:, self.common_features].copy()
        if scaler is not None and scaler.lower() in self.scalers:
            self.log(f"Using scaler '{scaler}' for the test data.")
            X_test = scale(X_test, self.scalers[scaler.lower()])
        self.n1_classifier.add_test_data(X_test)
        self.multi_classifier.add_test_data(X_test)
        self._test_data_added = True
        self.log("Test data added.")

    def fit(self, seed=0, on_n1_progress=None, on_multi_progress=None, n_jobs=1):
        if not self._test_data_added:
            raise RuntimeError("add_test_data() must be called before fit().")
        self.n1_classifier.fit(seed=seed, on_progress=on_n1_progress, n_jobs=n_jobs)
        self.multi_classifier.fit(seed=seed, on_progress=on_multi_progress, n_jobs=n_jobs)

    def predict_proba(self):
        """
        Blended probability estimates combining the N1 binary and multi-class classifiers:

        When the N1 vote gate fires (majority voted N1):
            p_final[N1]     = p_n1
            p_final[c≠N1]   = p_multi[c] * (1 - p_n1)
            → sums to 1; N1 is guaranteed winner since p_n1 ≥ 0.5

        When the N1 vote gate does not fire:
            p_final[N1]     = 0.0
            p_final[c≠N1]   = p_multi[c]
            → N1 is excluded; prediction = argmax(p_multi)

        p_n1 is the ensemble-mean N1 probability from the binary classifier.
        The gate itself is vote-based (consistent with predict()).
        """
        if not self._test_data_added:
            raise RuntimeError("add_test_data() and fit() must be called before predict_proba().")

        # Vote-based N1 gate; also populates n1_classifier.probs as a side-effect
        n1_is_positive = self.n1_classifier.predict()

        # p_n1: ensemble-mean probability that the sample is N1
        n1_probs_df = self.n1_classifier.probs
        true_col = next(c for c in n1_probs_df.columns if c is True or str(c) == "True")
        p_n1 = n1_probs_df[true_col]  # Series indexed by sample

        # Multi-class mean probabilities (trained on non-N1 samples only)
        multi_df = self.multi_classifier.predict_proba()
        non_n1_classes = [c for c in multi_df.columns if c != "N1"]

        final = pd.DataFrame(0.0, index=multi_df.index, columns=["N1"] + non_n1_classes)

        # Case 1: N1 gate fires — blend p_n1 with down-scaled multi-class probs
        if n1_is_positive.any():
            w = p_n1.loc[n1_is_positive].values          # shape (n_pos,)
            final.loc[n1_is_positive, "N1"] = w
            final.loc[n1_is_positive, non_n1_classes] = (
                multi_df.loc[n1_is_positive, non_n1_classes].values * (1 - w)[:, None]
            )

        # Case 2: N1 gate does not fire — exclude N1, use multi-class probs directly
        not_n1 = ~n1_is_positive
        if not_n1.any():
            final.loc[not_n1, non_n1_classes] = multi_df.loc[not_n1, non_n1_classes].values

        return final

    def predict(self):
        """
        Final predictions by ensemble vote:
        - N1 if a majority of N1-classifier models voted N1 (ties broken by mean probability)
        - Otherwise the multi-class label with the most votes
        """
        if not self._test_data_added:
            raise RuntimeError("add_test_data() and fit() must be called before predict().")

        # boolean Series: True where majority of N1-classifier models voted N1
        n1_is_positive = self.n1_classifier.predict()

        # label Series: plurality-vote winner among non-N1 classes
        multi_winner = self.multi_classifier.predict()

        preds = multi_winner.copy()
        preds.loc[n1_is_positive] = "N1"

        if self.verbose:
            self.log(f"Assigned N1 to {n1_is_positive.sum()} samples")

        return preds

    def export_results(self, path=None):
        preds = self.predict()

        results = pd.DataFrame(index=preds.index)
        results["Prediction"] = preds

        # --- N1 votes ---
        n1_votes = self.n1_classifier.all_votes
        results = results.join(n1_votes.add_prefix("N1_"), how="left")

        # --- multi-class votes ---
        multi_votes = self.multi_classifier.all_votes
        results = results.join(multi_votes.add_prefix("MULTI_"), how="left")
        if path is not None:
            results.to_csv(path)
        return results


def parse_features_per_model(_ctx, _param, value):
    """
    Parse the --features-per-model option.

    Accepts:
    - 'auto'  → returns the string 'auto'
    - integer strings like '200' → returns int(200)
    """
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return "auto"
        if v.isdigit():
            return int(v)

    raise click.BadParameter("features_per_model must be an integer or 'auto'.")


@click.command()
@click.argument(
    "data",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-m", "--models", default=100, type=click.IntRange(min=1), help="Number of models (default: 100)."
)
@click.option(
    "-f",
    "--features-per-model",
    default="auto",
    callback=parse_features_per_model,
    show_default=True,
    help=(
        "Number of features for each model, or 'auto' to choose based on "
        "expected coverage of 95% (default: 'auto')."
    ),
)
@click.option(
    "-s",
    "--scaler",
    default=RNASEQ_DGE,
    type=str,
    help=f"Scaler to use for the data. Options: '{RNASEQ_DGE}' (default), '{RNAPANEL}'.",
)
@click.option(
    "--no-scaler",
    is_flag=True,
    help="Disable scaling (set scaler to None).",
)
@click.option(
    "--ray/--no-ray",
    "use_ray",
    default=False,
    help="Use Ray for parallel training if installed.",
)
@click.option(
    "-r",
    "--seed",
    default=0,
    type=int,
    help="Random seed for reproducibility (default: 0).",
)
@click.option(
    "--out",
    default="results.csv",
    type=click.Path(dir_okay=False, writable=True),
    help="Name of the CSV file to save results.",
)
def classify_cli(
    data,
    models,
    features_per_model,
    scaler,
    no_scaler,
    use_ray,
    seed,
    out,
):
    if no_scaler:
        scaler = None
        logger.info("Scaling disabled, using input values.")
    else:
        scaler = scaler.lower()
        if scaler not in [RNASEQ_DGE, RNAPANEL]:
            logger.warning(
                f"Invalid scaler '{scaler}' specified. Using '{RNASEQ_DGE}' as default."
            )
            scaler = RNASEQ_DGE

    n_samples_per_model_n1 = 0.4

    if use_ray and ray is not None:
        print("Initializing Ray...")

    clf = LymphxpressClassifier(
        n_models=models,
        n_features_per_model=features_per_model,
        verbose=True,
        use_ray=use_ray,
        xgb_params_override_n1={"n_samples_per_model": n_samples_per_model_n1},
    )
    clf.add_test_data(pd.read_csv(data, index_col=0), scaler=scaler)
    clf.fit(seed=seed)  # no progress callbacks needed in CLI
    if out[-4:] != ".csv":
        out += ".csv"
    return clf.export_results(out)


if __name__ == "__main__":
    classify_cli()
