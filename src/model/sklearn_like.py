import copy
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMModel
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class BaseWrapper:
    model: Any
    name: str

    def fit(
        self,
        tr_x: NDArray,
        tr_y: NDArray,
        va_x: NDArray,
        va_y: NDArray,
        tr_w: NDArray | None = None,
    ) -> None:
        raise NotImplementedError

    def predict(self, X: NDArray) -> NDArray:  # noqa
        raise NotImplementedError

    def save(self, out_dir: str | Path) -> None:
        path = self.get_save_path(out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, out_dir: Path | str) -> None:
        path = self.get_save_path(out_dir)
        # debug in kaggle
        try:
            self.model = joblib.load(path)
        except Exception as e:
            print(e)
        self.fitted = True

    def get_save_path(self, out_dir: Path | str) -> Path:
        return Path(out_dir) / "model.pkl"

    def is_saved(self, out_dir: Path | str) -> bool:
        path = self.get_save_path(out_dir)
        return path.exists()


class LightGBMWapper(BaseWrapper):
    def __init__(
        self,
        name: str = "lgb",
        model: LGBMModel | None = None,
        fit_params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.name = name
        self.model = model or LGBMModel()
        self.fit_params = fit_params or {}
        self.fitted = False
        self.feature_names = feature_names or self.fit_params.get("feature_name")

        self.params = self.model.get_params()
        self.eval_metric = self.fit_params.get("eval_metric")

    def initialize(self) -> None:
        params = copy.deepcopy(self.params)
        self.model = LGBMModel(**params)

    def reshape_y(self, y: NDArray) -> NDArray:
        if y.ndim == 1:
            return y
        if y.shape[1] == 1:
            return y.reshape(-1)
        return y

    def fit(
        self,
        tr_x: NDArray,
        tr_y: NDArray,
        va_x: NDArray,
        va_y: NDArray,
        tr_w: NDArray | None = None,
    ) -> None:
        self.initialize()
        self.model.fit(
            tr_x,
            self.reshape_y(tr_y),
            eval_set=[(va_x, self.reshape_y(va_y))],
            sample_weight=list(tr_w) if tr_w is not None else None,
            **self.fit_params,
        )
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if not self.fitted:
            raise ValueError("Model is not fitted yet")
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> Any:
        return self.model.feature_importances_


class XGBoostRegressorWrapper(BaseWrapper):
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    def __init__(
        self,
        name: str = "xgb",
        model: XGBRegressor | None = None,
        fit_params: dict[str, Any] | None = None,
        early_stopping_params: dict[str, Any] | None = None,  # callbacks を引き継いでしまうので注意
        cat_features: list[int] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.name = name
        self.model = model or XGBRegressor()
        self.fit_params = fit_params or {}
        self.fitted = False
        self.early_stopping_params = early_stopping_params or {}
        self.cat_features = cat_features
        self.feature_names = feature_names

        self.params = self.model.get_params()
        self.eval_metric = self.params.get("eval_metric")

    def initialize(self) -> None:
        params = copy.deepcopy(self.params)
        params["eval_metric"] = self.eval_metric

        if self.early_stopping_params:
            # 同じ instance だとバグるので initialize する
            callbacks = params.get("callbacks") or []
            early_stopping_callback = xgb.callback.EarlyStopping(**self.early_stopping_params)
            params["callbacks"] = callbacks + [early_stopping_callback]

        self.model = XGBRegressor(**params)

    def fit(
        self,
        tr_x: NDArray,
        tr_y: NDArray,
        va_x: NDArray,
        va_y: NDArray,
        tr_w: NDArray | None = None,
    ) -> None:
        self.initialize()

        if self.cat_features:
            tr_x = (
                pl.DataFrame(tr_x, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
                .astype({x: "category" for x in self.cat_features})
            )
            va_x = (
                pl.DataFrame(va_x, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
                .astype({x: "category" for x in self.cat_features})
            )
        self.model.fit(
            tr_x,
            tr_y,
            eval_set=[(va_x, va_y)],
            sample_weight=tr_w,
            **self.fit_params,
        )
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        if self.cat_features:
            X = (
                pl.DataFrame(X, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
                .astype({x: "category" for x in self.cat_features})
            )
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> Any:
        return self.model.feature_importances_


class XGBoostClassifierWrapper(XGBoostRegressorWrapper):
    def initialize(self) -> None:
        params = copy.deepcopy(self.params)
        params["eval_metric"] = self.eval_metric

        if self.early_stopping_params:
            callbacks = params.get("callbacks") or []
            early_stopping_callback = xgb.callback.EarlyStopping(**self.early_stopping_params)
            params["callbacks"] = callbacks + [early_stopping_callback]

        self.model = xgb.XGBClassifier(**params)

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        if self.cat_features:
            X = (
                pl.DataFrame(X, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )
        return self.model.predict_proba(X)[:, 1]


class CatBoostRegressorWrapper(BaseWrapper):
    # https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
    def __init__(
        self,
        name: str = "cat",
        model: CatBoostRegressor | None = None,
        fit_params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        cat_features: list[int] | None = None,
        multi_output: bool = False,
    ):
        self.name = name
        self.model = model or CatBoostRegressor()
        self.fit_params = fit_params or {}
        self.fitted = False
        self.feature_names = feature_names
        self.cat_features = cat_features
        if self.cat_features:
            self.fit_params["cat_features"] = self.cat_features

        self.model.set_feature_names(feature_names)
        self.eval_metric = self.model.get_params().get("eval_metric")

        self.loss_function = self.model.get_params().get("loss_function")
        self.multi_output = multi_output

    def initialize(self) -> None:
        params = copy.deepcopy(self.model.get_params())
        params["eval_metric"] = self.eval_metric
        self.model = CatBoostRegressor(**params)

    def fit(
        self,
        tr_x: NDArray,
        tr_y: NDArray,
        va_x: NDArray,
        va_y: NDArray,
        tr_w: NDArray | None = None,
    ) -> None:
        self.initialize()
        if self.cat_features:
            tr_x = (
                pl.DataFrame(tr_x, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )
            va_x = (
                pl.DataFrame(va_x, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )

        self.model.fit(
            tr_x,
            tr_y,
            eval_set=(va_x, va_y),
            sample_weight=tr_w,
            **self.fit_params,
        )
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if self.cat_features:
            X = (
                pl.DataFrame(X, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )

        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        preds = self.model.predict(X)

        if self.loss_function.endswith("WithUncertainty"):
            return preds[:, 0]

        if self.multi_output:
            return preds

        if preds.ndim > 1 and not self.multi_output:
            return preds[:, 0]  # return only the first column

        return preds

    @property
    def feature_importances_(self) -> Any:
        return self.model.get_feature_importance()


class CatBoostClassifierWrapper(CatBoostRegressorWrapper):
    def initialize(self) -> None:
        params = copy.deepcopy(self.model.get_params())
        params["eval_metric"] = self.eval_metric
        self.model = CatBoostClassifier(**params)

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if self.cat_features:
            X = (
                pl.DataFrame(X, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )

        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        preds = self.model.predict_proba(X)

        if preds.ndim == 2:
            return preds[:, 1]  # return positive class probability if binary classification

        return preds


class LinearWrapper(BaseWrapper):
    def __init__(
        self,
        name: str = "linear",
        model: Ridge | None = None,
        feature_names: list[str] | None = None,
        scaling: bool = False,
        impute: bool = False,
    ):
        self.name = name
        self.model = model
        self.fitted = False
        self.feature_names = feature_names
        self.scaling = scaling
        self.impute = impute
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")

    def fit(
        self,
        tr_x: NDArray,
        tr_y: NDArray,
        va_x: NDArray,
        va_y: NDArray,
        tr_w: NDArray | None = None,
    ) -> None:
        # Handle imputation if enabled
        if self.impute:
            tr_x = self.imputer.fit_transform(tr_x)
            va_x = self.imputer.transform(va_x)

        # Handle scaling if enabled
        if self.scaling:
            tr_x = self.scaler.fit_transform(tr_x)
            va_x = self.scaler.transform(va_x)

        if tr_w is not None:
            self.model.fit(tr_x, tr_y, sample_weight=tr_w)
        else:
            self.model.fit(tr_x, tr_y)
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        # Copy X to avoid modifying the original data
        X_processed = X.copy()

        # Apply imputation if enabled
        if self.impute:
            X_processed = self.imputer.transform(X_processed)

        # Apply scaling if enabled
        if self.scaling:
            X_processed = self.scaler.transform(X_processed)

        return self.model.predict(X_processed).reshape(-1)

    @property
    def feature_importances_(self) -> Any:
        # output coef_ as feature_importances_
        return self.model.coef_.reshape(-1)

    def save(self, out_dir: str | Path) -> None:
        path = self.get_save_path(out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model along with preprocessors
        data_to_save = {"model": self.model}

        if self.scaling:
            data_to_save["scaler"] = self.scaler

        if self.impute:
            data_to_save["imputer"] = self.imputer

        joblib.dump(data_to_save, path)

    def load(self, out_dir: Path | str) -> None:
        path = self.get_save_path(out_dir)
        # debug in kaggle
        try:
            data = joblib.load(path)

            # Check if the saved object is a dict or direct model
            if isinstance(data, dict):
                self.model = data["model"]

                if self.scaling and "scaler" in data:
                    self.scaler = data["scaler"]

                if self.impute and "imputer" in data:
                    self.imputer = data["imputer"]
            else:
                # For backward compatibility with old saved models
                self.model = data
        except Exception as e:
            print(e)
        self.fitted = True
