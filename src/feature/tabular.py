import abc
from collections.abc import Iterable
from itertools import combinations
from typing import Any

import category_encoders as ce
import numpy as np
import polars as pl
from sklearn.model_selection import BaseCrossValidator, KFold
from tqdm import tqdm


class BaseEncoder:
    fitted: bool
    prefix: str

    @abc.abstractmethod
    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


class OneHotEncoder(BaseEncoder):
    def __init__(self, columns: list[str], prefix: str = "f_") -> None:
        self.columns = columns
        self.fitted = False
        self.prefix = prefix

        self.encoder = ce.OneHotEncoder()

    def fit(self, df: pl.DataFrame) -> None:
        self.encoder.fit(df.select(self.columns).to_pandas())
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")

        df = self.encoder.transform(df.select(self.columns).to_pandas())
        return pl.DataFrame(df).select(pl.all().name.prefix(self.prefix))


class OrdinalEncoder(BaseEncoder):
    def __init__(self, columns: list[str], prefix: str = "f_") -> None:
        self.columns = columns
        self.fitted = False
        self.prefix = prefix

        self.encoder = ce.OrdinalEncoder(handle_missing="return_nan", handle_unknown="return_nan")

    def fit(self, df: pl.DataFrame) -> None:
        self.encoder.fit(df.select(self.columns).to_pandas())
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")

        return pl.DataFrame(
            self.encoder.transform(df.select(self.columns).to_pandas()),
            schema=self.columns,
        ).select(pl.all().name.prefix(self.prefix))


class AggregateEncoder(BaseEncoder):
    def __init__(
        self,
        group_keys: list[str] | str,
        agg_exprs: list[pl.Expr],
        prefix: str = "f_",
        extra_transforms: list[dict[str, dict[str, str]]] | None = None,
    ) -> None:
        """aggregate numerical columns by group keys

        Args:
            group_keys (list[str] | str): list or single string of column names to group by
            agg_exprs (list[pl.Expr]): list of aggregation expressions (e.g. pl.col("colname").mean().alias("mean_colname"))
                - set alias as the output column name
                - `agg_{expr_alias}_{group_key_name}` will be used as the output column name
            prefix (str, optional): prefix of the output column name. Defaults to "f_".
            extra_transforms (list[dict[str, dict[str, str]]] | None, optional): extra transformation methods to apply. Defaults to None.

        Example:
        ```
        encoder = AggregateEncoder(
            group_keys="category",
            agg_exprs=[pl.col("val").mean().alias("mean_val"), pl.col("val").std().alias("std_val")],
            prefix="f_",
            extra_transforms=[
                {
                    "method": "z-score",
                    "val": "val",
                    "mean": "mean_val",
                    "std": "std_val",
                },
            ],
        )
        ```
        """
        self.group_keys = group_keys
        self.agg_exprs = agg_exprs
        self.prefix = prefix
        self.extra_transforms = extra_transforms or []
        self.mapping_df: pl.DataFrame | None = None
        self.fitted = False

    @property
    def group_key_name(self) -> str:
        if isinstance(self.group_keys, str):
            return self.group_keys
        return "_x_".join(self.group_keys)

    @property
    def aggregated_colnames(self) -> dict[str, str]:
        return {name: f"{name}_grpby_{self.group_key_name}" for name in self.expr_output_names}

    @property
    def expr_output_names(self) -> list[str]:
        return [expr.meta.output_name() for expr in self.agg_exprs]

    def fit(self, df: pl.DataFrame) -> None:
        agg_exprs = self.agg_exprs.copy()
        first_expr = agg_exprs.pop(0)
        self.mapping_df = df.group_by(self.group_keys, maintain_order=True).agg(first_expr)
        for expr in agg_exprs:
            print(f"processing: {expr}")
            self.mapping_df = self.mapping_df.join(
                df.group_by(self.group_keys, maintain_order=True).agg(expr),
                on=self.group_keys,
                how="left",
            )

        self.mapping_df = self.mapping_df.rename(self.aggregated_colnames)
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")

        base_df = df.join(self.mapping_df, on=self.group_keys, how="left")
        if not self.extra_transforms:
            return base_df.select(list(self.aggregated_colnames.values())).select(pl.all().name.prefix(self.prefix))

        out_cols = list(self.aggregated_colnames.values())
        for extra_transform in self.extra_transforms:
            method = extra_transform.pop("method")
            _agg_names = extra_transform.copy()
            if method == "z-score":
                assert sorted(list(_agg_names.keys())) == sorted(["val", "mean", "std"])
                extra_colname = f"{method}_{_agg_names['val']}_grpby_{self.group_key_name}"
                base_df = base_df.with_columns(
                    (
                        (pl.col(_agg_names["val"]) - pl.col(self.aggregated_colnames[_agg_names["mean"]]))
                        / pl.col(self.aggregated_colnames[_agg_names["std"]])
                    ).alias(extra_colname)
                )
                out_cols.append(extra_colname)
            elif method == "med-abs-dev":
                assert sorted(list(_agg_names.keys())) == sorted(["val", "median"])
                extra_colname = f"{method}_{_agg_names['val']}_grpby_{self.group_key_name}"
                base_df = base_df.with_columns(
                    (pl.col(_agg_names["val"]) - pl.col(self.aggregated_colnames[_agg_names["median"]]))
                    .abs()
                    .alias(extra_colname)
                )
                out_cols.append(extra_colname)

            elif method == "mean-diff":
                assert sorted(list(_agg_names.keys())) == sorted(["val", "mean"])
                extra_colname = f"{method}_{_agg_names['val']}_grpby_{self.group_key_name}"
                base_df = base_df.with_columns(
                    (pl.col(_agg_names["val"]) - pl.col(self.aggregated_colnames[_agg_names["mean"]])).alias(
                        extra_colname
                    )
                )
                out_cols.append(extra_colname)

            elif method == "median-diff":
                assert sorted(list(_agg_names.keys())) == sorted(["val", "median"])
                extra_colname = f"{method}_{_agg_names['val']}_grpby_{self.group_key_name}"
                base_df = base_df.with_columns(
                    (pl.col(_agg_names["val"]) - pl.col(self.aggregated_colnames[_agg_names["median"]])).alias(
                        extra_colname
                    )
                )
                out_cols.append(extra_colname)

            elif method == "mean-ratio":
                assert sorted(list(_agg_names.keys())) == sorted(["val", "mean"])
                extra_colname = f"{method}_{_agg_names['val']}_grpby_{self.group_key_name}"
                base_df = base_df.with_columns(
                    (pl.col(_agg_names["val"]) / pl.col(self.aggregated_colnames[_agg_names["mean"]])).alias(
                        extra_colname
                    )
                )
                out_cols.append(extra_colname)

            elif method == "range":  # (max- min)
                assert sorted(list(_agg_names.keys())) == sorted(["min", "max"])
                extra_colname = f"{method}_{_agg_names['max']}_{_agg_names['min']}_grpby_{self.group_key_name}"
                base_df = base_df.with_columns(
                    (pl.col(_agg_names["max"]) - pl.col(_agg_names["min"])).alias(extra_colname)
                )
                out_cols.append(extra_colname)

        return base_df.select(out_cols).select(pl.all().name.prefix(self.prefix))

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


class RawEncoder(BaseEncoder):
    def __init__(self, columns: list[str], prefix: str = "f_") -> None:
        self.columns = columns
        self.fitted = False
        self.prefix = prefix

    def fit(self, df: pl.DataFrame) -> None:
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")
        return df.select(self.columns).select(pl.all().name.prefix(self.prefix))


class NullFlagEncoder(BaseEncoder):
    def __init__(
        self,
        columns: list[str] | None = None,
        prefix: str = "f_",
        suffix: str = "_is_null",
    ) -> None:
        self.columns = columns or []
        self.fitted = False
        self.prefix = prefix

        self.suffix = suffix

    def fit(self, df: pl.DataFrame) -> None:
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(self.columns) == 0:
            self.columns = df.columns

        return (
            df.select(pl.col(self.columns).is_null())
            .select(pl.all().name.prefix(self.prefix))
            .select(pl.all().name.prefix(self.suffix))
        )


class ArithmeticCombinationEncoder(BaseEncoder):
    """Calculate features by arithmetic combinations of numerical columns using polars."""

    def __init__(
        self,
        input_cols=None,
        include_cols=None,
        exclude_cols=None,
        operator="+",
        output_prefix="",
        output_suffix="_combi",
        drop_origin=False,
        r=2,
    ):
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._exclude_cols = exclude_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._r = r
        self._operator = operator
        self._drop_origin = drop_origin

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit to data frame, then transform it.

        Args:
            df (pl.DataFrame): Input data frame.
        Returns:
            pl.DataFrame: Output data frame.
        """
        new_df = df.clone()

        if not self._input_cols:
            self._input_cols = [col for col in new_df.columns if col not in self._exclude_cols]

        return self.transform(new_df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data frame.

        Args:
            df (pl.DataFrame): Input data frame.
        """
        new_df = df.clone()
        generated_cols = []

        n_fixed_cols = len(self._include_cols)

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            fixed_cols_str = f"{self._operator}".join(self._include_cols)
            pairs_cols_str = f"{self._operator}".join(sorted(list(cols_pairs)))
            new_col = (
                self._output_prefix
                + fixed_cols_str
                + (self._operator if fixed_cols_str else "")
                + pairs_cols_str
                + self._output_suffix
            )
            if new_col in generated_cols:
                continue

            concat_cols = self._include_cols + list(cols_pairs)

            # Perform arithmetic operation
            if self._operator == "+":
                new_df = new_df.with_columns(pl.sum_horizontal(*concat_cols).alias(new_col))
            elif self._operator == "*":
                arr = new_df.select(concat_cols).to_numpy()
                new_df = new_df.with_columns(pl.Series(np.prod(arr, axis=1)).alias(new_col))
            else:
                raise ValueError(f"Operator {self._operator} is not supported.")

            generated_cols.append(new_col)

        if self._drop_origin:
            return new_df.select(generated_cols)

        return new_df


class StringCombinationEncoder(BaseEncoder):
    """Create new features by concatenating string columns using polars."""

    def __init__(
        self,
        input_cols=None,
        include_cols=None,
        exclude_cols=None,
        separator=" ",
        output_prefix="",
        output_suffix="_concat",
        drop_origin=False,
        r=2,
    ):
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._exclude_cols = exclude_cols or []
        self._separator = separator
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._drop_origin = drop_origin
        self._r = r

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit to data frame, then transform it.

        Args:
            df (pl.DataFrame): Input data frame.
        Returns:
            pl.DataFrame: Output data frame.
        """
        new_df = df.clone()

        if not self._input_cols:
            self._input_cols = [
                col for col in new_df.columns if col not in self._exclude_cols and new_df[col].dtype == pl.Utf8
            ]

        return self.transform(new_df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data frame.

        Args:
            df (pl.DataFrame): Input data frame.
        """
        new_df = df.clone()
        generated_cols = []

        n_fixed_cols = len(self._include_cols)

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            concat_cols = self._include_cols + list(cols_pairs)
            cols_str = self._separator.join(concat_cols)
            new_col = self._output_prefix + cols_str + self._output_suffix
            if new_col in generated_cols:
                continue

            # Perform string concatenation
            new_df = new_df.with_columns(pl.concat_str(concat_cols, separator=self._separator).alias(new_col))

            generated_cols.append(new_col)

        if self._drop_origin:
            return new_df.select(generated_cols)

        return new_df


class CatColumnCombinations:
    new_cols = set()

    def __call__(self, input_df: pl.DataFrame, include_cols: list[str], r: int = 2) -> pl.DataFrame:
        # Validate inputs
        if not include_cols:
            self.new_cols = list(self.new_cols)
            return input_df

        if not include_cols or r <= 0 or r > len(include_cols):
            raise ValueError("Invalid combination parameters.")

        output_df = input_df.clone()

        # Generate all combinations of columns
        col_combinations = list(combinations(include_cols, r))

        # Add each combination as a new column in the DataFrame
        for comb in tqdm(col_combinations):
            new_col = "+".join(comb)
            output_df = output_df.with_columns(
                pl.concat_str([pl.col(x).fill_null("NULL").cast(pl.String) for x in comb]).alias(new_col)
            )
            self.new_cols.add(new_col)

        self.new_cols = sorted(list(self.new_cols))
        return output_df


class TargetEncoder(BaseEncoder):
    def __init__(
        self,
        x_columns: list[str],
        y_column: str,
        folds: Iterable | BaseCrossValidator | None = None,
        folds_params: dict[str, Any] | None = None,
        smoothing_method: str = "none",
        smoothing_params: dict[str, Any] | None = None,
        prefix: str = "f_",
    ) -> None:
        import shirokumas as sk

        sk._target._UNKNOWN_VALUE = -1.0
        sk._target._MISSING_VALUE = -2.0

        self.x_columns = x_columns
        self.y_column = y_column
        self.fitted = False
        self.prefix = prefix

        if folds is None:
            folds = KFold(n_splits=5, shuffle=True, random_state=42)

        self.encoder = sk.TargetEncoder(
            folds=folds,
            folds_params=folds_params,
            smoothing_method=smoothing_method,
            smoothing_params=smoothing_params,
        )

    def fit(self, df: pl.DataFrame) -> None:
        self.encoder.fit(X=df.select(self.x_columns), y=df[self.y_column])
        self.fitted = True

    @property
    def new_colname_mapping(self):
        mapping = {col: f"{self.prefix}te_by_{self.y_column}_{col}" for col in self.x_columns}
        return mapping

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")

        out_df = self.encoder.transform(df.select(self.x_columns))
        out_df = out_df.rename(self.new_colname_mapping)

        return out_df
