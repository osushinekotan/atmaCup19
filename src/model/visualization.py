import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

sns.set_style("whitegrid")
japanize_matplotlib.japanize()


def plot_feature_importance(
    df: pd.DataFrame,
    feature_name_col: str = "feature_name",
    feature_importance_col: str = "feature_importance",
    fold_col: str = "fold",
    top_k: int = None,
) -> plt.Figure:
    # Determine if it's a single fold or multiple folds
    if fold_col not in df.columns:
        is_single_fold = True
    else:
        is_single_fold = df[fold_col].nunique() == 1

    # Filter the top_k features if specified
    if top_k is not None:
        # Compute mean importance to determine the top_k features
        df_mean = df.groupby(feature_name_col, as_index=False)[feature_importance_col].mean()
        top_features = df_mean.nlargest(top_k, feature_importance_col)[feature_name_col].tolist()
        df = df[df[feature_name_col].isin(top_features)]

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))
    if is_single_fold:
        # Barplot for a single fold
        df_mean = df.groupby(feature_name_col, as_index=False)[feature_importance_col].mean()
        df_mean = df_mean.sort_values(by=feature_importance_col, ascending=False)

        sns.barplot(
            data=df_mean,
            x=feature_importance_col,
            y=feature_name_col,
            orient="h",
            ax=ax,
        )
        ax.set_title("Feature Importance (Single Fold)", fontsize=16)
    else:
        # Boxplot for multiple folds
        df_sorted = df.groupby(feature_name_col, as_index=False)[feature_importance_col].mean()
        feature_order = df_sorted.sort_values(by=feature_importance_col, ascending=False)[feature_name_col]

        sns.boxplot(
            data=df,
            x=feature_importance_col,
            y=feature_name_col,
            order=feature_order,
            ax=ax,
        )
        ax.set_title("Feature Importance Across Folds", fontsize=16)

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig


def visualize_prediction_errors(
    df: pl.DataFrame | pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    index_col: str | None = None,
    figsize: tuple[int, int] = (18, 14),
    bins: int = 30,
    save_path: str | None = None,
    title_prefix: str = "",
) -> None:
    if isinstance(df, pd.DataFrame):
        pdf = df
    elif isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()

    # 誤差計算
    pdf["error"] = pdf[y_true_col] - pdf[y_pred_col]
    pdf["abs_error"] = np.abs(pdf["error"])
    pdf["perc_error"] = 100 * pdf["error"] / pdf[y_true_col].where(pdf[y_true_col] != 0, 1e-10)

    # 誤差のヒストグラム、分布と散布図をプロット
    plt.figure(figsize=figsize)

    # 1. 誤差のヒストグラム
    ax1 = plt.subplot(3, 2, 1)
    sns.histplot(pdf["error"], bins=bins, kde=True, ax=ax1)
    ax1.set_title(f"{title_prefix}予測誤差の分布")
    ax1.set_xlabel("誤差 (y_true - y_pred)")
    ax1.axvline(0, color="r", linestyle="--")

    # 2. 絶対誤差のヒストグラム
    ax2 = plt.subplot(3, 2, 2)
    sns.histplot(pdf["abs_error"], bins=bins, kde=True, ax=ax2)
    ax2.set_title(f"{title_prefix}絶対誤差の分布")
    ax2.set_xlabel("絶対誤差 |y_true - y_pred|")

    # 3. 実測値と予測値の散布図
    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(pdf[y_true_col], pdf[y_pred_col], alpha=0.5)

    # 理想線（y=x）を追加
    min_val = min(pdf[y_true_col].min(), pdf[y_pred_col].min())
    max_val = max(pdf[y_true_col].max(), pdf[y_pred_col].max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--")

    ax3.set_title(f"{title_prefix}実測値 vs 予測値")
    ax3.set_xlabel("実測値 (y_true)")
    ax3.set_ylabel("予測値 (y_pred)")

    # 4. 実測値に対する誤差の散布図
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(pdf[y_true_col], pdf["error"], alpha=0.5)
    ax4.axhline(0, color="r", linestyle="--")
    ax4.set_title(f"{title_prefix}実測値に対する誤差")
    ax4.set_xlabel("実測値 (y_true)")
    ax4.set_ylabel("誤差")

    # 5. 実測値と予測値の分布比較
    ax5 = plt.subplot(3, 2, 5)
    sns.kdeplot(pdf[y_true_col], ax=ax5, label="実測値")
    sns.kdeplot(pdf[y_pred_col], ax=ax5, label="予測値")
    ax5.set_title(f"{title_prefix}実測値と予測値の分布比較")
    ax5.set_xlabel("値")
    ax5.legend()

    # 6. 誤差の箱ひげ図
    ax6 = plt.subplot(3, 2, 6)
    sns.boxplot(y=pdf["error"], ax=ax6)
    ax6.set_title(f"{title_prefix}誤差の箱ひげ図")
    ax6.set_ylabel("誤差")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
