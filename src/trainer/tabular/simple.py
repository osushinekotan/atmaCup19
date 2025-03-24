import copy
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.model.sklearn_like import BaseWrapper
from src.model.visualization import plot_feature_importance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def single_train_fn(  # noqa: C901
    model: BaseWrapper,
    features_df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str | list[str],
    meta_cols: list[str],
    weight_col: str | None = None,
    fold_col: str | Path = "fold",
    out_dir: str | Path = "./outputs",
    eval_fn: None = None,
    valid_folds: list[int] | None = None,
    overwrite: bool = False,
    pred_col_name: str = "pred",
    val_features_df: pl.DataFrame | None = None,
    full_training: bool = False,
    **kwargs,
) -> tuple[pl.DataFrame, dict[str, float], list[BaseWrapper]]:
    if (val_features_df is None) and full_training:
        raise ValueError("val_features_df must be specified when full_training is True")

    va_records, va_scores, trained_models = [], {}, []
    out_dir = Path(out_dir) / model.name

    valid_folds = valid_folds or features_df[fold_col].unique().to_list()
    valid_folds = n_valid_folds if (n_valid_folds := len(valid_folds)) > 1 else [min(valid_folds) - 1]  # holdout

    use_eval_metric_extra_va_df = kwargs.get("use_eval_metric_extra_va_df", False)
    enable_plot_feature_importance = kwargs.get("enable_plot_feature_importance", True)

    for i_fold in valid_folds:
        logger.info(f"🚀 >>> Start training fold {i_fold} =============")
        tr_x = features_df.filter(pl.col(fold_col) != i_fold).select(feature_cols).to_numpy()
        tr_y = features_df.filter(pl.col(fold_col) != i_fold)[target_col].to_numpy()

        # 明示的に指定された validation data がある場合はそちらを使う
        va_df = features_df.filter(pl.col(fold_col) == i_fold) if val_features_df is None else val_features_df
        va_x = va_df.select(feature_cols).to_numpy()
        va_y = va_df[target_col].to_numpy()

        tr_w = features_df.filter(pl.col(fold_col) != i_fold)[weight_col].to_numpy() if weight_col else None

        i_out_dir = out_dir / f"fold_{i_fold:02}"

        if use_eval_metric_extra_va_df:
            model.eval_metric.va_df = va_df

        if model.get_save_path(out_dir=i_out_dir).exists() and not overwrite:
            model.load(out_dir=i_out_dir)
            logger.info(f"   - ❌ Skip training fold {i_fold}")
        else:
            model.fit(tr_x=tr_x, tr_y=tr_y, va_x=va_x, va_y=va_y, tr_w=tr_w)
            if not full_training:
                model.save(out_dir=i_out_dir)

        va_pred = model.predict(va_x)

        if full_training:
            logger.info("   - 🚀 >>> Start full training")
            full_df = pl.concat([features_df, val_features_df], how="diagonal_relaxed")
            tr_x_full = full_df.select(feature_cols).to_numpy()
            tr_y_full = full_df[target_col].to_numpy()
            tr_w_full = full_df[weight_col].to_numpy() if weight_col else None
            model.fit(tr_x=tr_x_full, tr_y=tr_y_full, tr_w=tr_w_full)  # NOTE: update best iter
            model.save(out_dir=i_out_dir)

        trained_models.append(copy.deepcopy(model))
        logger.info("   - ✅ Successfully saved model")

        try:
            if va_pred.shape[1] == 1:
                va_pred = va_pred.reshape(-1)
        except IndexError:
            pass

        i_va_df = va_df.select([c for c in meta_cols if c in va_df.columns]).with_columns(
            [
                pl.Series(va_pred).alias(pred_col_name),
                pl.lit(model.name).alias("name"),
            ]
        )
        i_va_df.write_parquet(i_out_dir / "va_pred.parquet")
        va_records.append(i_va_df)
        logger.info("   - ✅ Successfully predicted validation data")

        i_score = eval_fn(input_df=i_va_df)
        va_scores[f"{eval_fn.__name__}_fold_{i_fold:02}"] = i_score

        logger.info(f"   - {eval_fn.__name__}: {i_score}")
        logger.info(f"🎉 <<< Finish training fold {i_fold} =============\n\n")

    va_result_df = pl.concat(va_records, how="diagonal")
    score = eval_fn(input_df=va_result_df)
    va_scores[f"{eval_fn.__name__}_full"] = score
    logger.info(f"✅ Final {eval_fn.__name__}: {score}")

    # plot feature importance
    if enable_plot_feature_importance:
        import pandas as pd

        importance_df = pd.DataFrame()
        for i, m in enumerate(trained_models):
            i_df = pd.DataFrame(
                {"feature_importance": m.feature_importances_, "feature_name": m.feature_names, "fold": i}
            )
            importance_df = pd.concat([importance_df, i_df], axis=0, ignore_index=True)

        fig = plot_feature_importance(
            df=importance_df,
            feature_name_col="feature_name",
            feature_importance_col="feature_importance",
            fold_col="fold",
            top_k=50,
        )
        fig.savefig(out_dir / "feature_importance.png", dpi=300)

    with open(out_dir / "va_scores.json", "w") as f:
        json.dump(va_scores, f, indent=4)

    va_result_df.write_parquet(out_dir / "va_result.parquet")

    return va_result_df, va_scores, trained_models


def single_inference_fn_v2(
    models: list[BaseWrapper],
    features_df: pl.DataFrame,
    feature_names: list[str],
) -> pl.DataFrame:
    te_preds = []

    te_x = features_df.select(feature_names).to_numpy()
    try:
        for model in models:
            te_pred = model.predict(te_x)
            try:
                if te_pred.shape[1] == 1:
                    te_pred = te_pred.reshape(-1)
            except IndexError:
                pass
            te_preds.append(te_pred)
    except Exception as e:
        logger.error(f"   - ❌ Failed to load model: {e}")

    te_pred_ave = np.mean(te_preds, axis=0)

    # Save prediction
    excepted_features = [f for f in features_df.columns if f not in feature_names]
    te_result_df = features_df.select(excepted_features).with_columns(pl.Series("pred", te_pred_ave))

    return te_result_df


def single_inference_fn(
    model: BaseWrapper,
    features_df: pl.DataFrame,
    feature_names: list[str],
    model_dir: str | Path,
    inference_folds: list[int],
    out_dir: str | Path = "./outputs",
) -> pl.DataFrame:
    te_preds = []
    model_dir = Path(model_dir) / model.name

    out_dir = Path(out_dir) / model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i_fold in inference_folds:
        logger.info(f"🚀 >>> Start training fold {i_fold} =============")

        te_x = features_df.select(feature_names).to_numpy()
        i_out_dir = Path(model_dir) / f"fold_{i_fold:02}"

        try:
            model.load(out_dir=i_out_dir)
            logger.info("   - ✅ Successfully loaded model")

            te_pred = model.predict(te_x)
            try:
                if te_pred.shape[1] == 1:
                    te_pred = te_pred.reshape(-1)
            except IndexError:
                pass
            te_preds.append(te_pred)
        except Exception as e:
            logger.error(f"   - ❌ Failed to load model: {e}")

    te_pred_ave = np.mean(te_preds, axis=0)

    # Save prediction
    excepted_features = [f for f in features_df.columns if f not in feature_names]
    te_result_df = features_df.select(excepted_features).with_columns(pl.Series("pred", te_pred_ave))
    te_result_df.write_parquet(out_dir / "te_result.parquet")

    return te_result_df
