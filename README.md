# atmaCup19

## validation

「[より"テスト"っぽい検証方法](https://www.guruguru.science/competitions/26/discussions/d4ddc671-1c92-4d9d-9e21-41ffff7df100/)」を参考に holdout で検証用データを作成。

ただし `test_session.csv` の `顧客CD` はユニークであることに注意して、以下のように検証データを選定

- `train_session.csv` における最新の session のに限定
- 10 月中の session に限定

```python
valid_session_ids = (
    train_session_df.sort("売上日")
    .group_by("顧客CD")
    .tail(1)
    .filter(pl.col("売上日") >= VALID_DATE)["session_id"]
    .unique()
    .sample(fraction=VALID_SAMPLE_FRAC, seed=SEED)
)
valid_session_df = train_session_df.filter(pl.col("session_id").is_in(valid_session_ids))
train_session_df = train_session_df.filter(pl.col("session_id").is_in(valid_session_ids).not_())
```

## Feature Engineering

1. 商品カテゴリを買った日から何日後の session か特徴量
2. rolling 集約系の特徴量
3. rolling 集約 -> svd 特徴量
4. online target encoding 特徴量
5. 日付や年代などの特徴量

- 合計 1000~2000 程度の特徴量を生成
- リークしないように特徴生成
- カテゴリが多すぎる場合は、共起数や同部門/ディビジョンのカテゴリなどでフィルタリング

### online target encoding

- colum2131 さんの特徴量
- リークに注意しながら baysian target encoding を session 単位で更新するイメージ
- polars で 1 line で書ける

```python
def create_online_target_encoded_df(
    session_df: pl.DataFrame,
    jan_df: pl.DataFrame,
    train_log_df: pl.DataFrame,
    target_categories: list[str],
    target_value: str = "売上有無",
    cat_type: str = "カテゴリ名",
    group_col: str = "顧客CD",
    mean_group_cols: list[str] = ["性別", "年代"],  # noqa
    alpha: float = 1.0,
    prefix: str = "online_target_encoded_",
) -> pl.DataFrame:
    result_df = (
        (
            session_df.select("session_id")
            .join(train_log_df, on="session_id", how="left")
            .join(jan_df.filter(pl.col(cat_type).is_in(target_categories)), on="JAN", how="left")
            .select(["session_id", cat_type, "売上数量"])
            .group_by(["session_id", cat_type])
            .agg(pl.col("売上数量").sum())
            .filter(pl.col("売上数量") > 0)
            .with_columns(pl.lit(1).alias("売上有無"))
            # columns: cat_type, index: session_id, values: 売上有無 の cross tab
            .pivot(cat_type, index="session_id", values=target_value)
            .select(["session_id"] + target_categories)
            # 形式を合わせるために session df を right join
            .join(session_df, on="session_id", how="right")
            .select(
                [pl.col("session_id"), pl.col("session_datetime"), pl.col(group_col)]
                + mean_group_cols
                + [pl.col(x).fill_null(0) for x in target_categories]
            )
            .sort([group_col, "session_datetime"])
        )
        # full record を使った平均の計算
        .with_columns(pl.col(cat).mean().over(mean_group_cols).alias(f"{cat}_mean") for cat in target_categories)
        # 自身のレコードを除外した累積和の計算
        .with_columns(
            pl.col(cat).cum_sum().over(group_col).shift(1).fill_null(0).alias(f"{cat}_cumsum_exclude_self")
            for cat in target_categories
        )
        # 自身のレコードを除外したカウントの計算
        .with_columns(
            pl.cum_count(group_col).over(group_col).shift(1).fill_null(0).alias(f"{group_col}_cumcount_exclude_self")
        )
        # online target encoding の計算
        .with_columns(
            (
                (pl.col(f"{cat}_cumsum_exclude_self") + pl.col(f"{cat}_mean") * alpha)
                / (pl.col(f"{group_col}_cumcount_exclude_self") + alpha)
            ).alias(f"{prefix}_{cat}")
            for cat in target_categories
        )
        .select(["session_id"] + [f"{prefix}_{cat}" for cat in target_categories])
    )

    return result_df
```

## Model

lightgbm (full training)

- test データ予測に使用するデータは、すべての期間の学習データを使用
- validation 時の best iteration をそのまま使用した
- public score: `0.7821`
