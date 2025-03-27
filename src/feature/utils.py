import functools
import hashlib
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar, cast, overload

import joblib
import polars as pl

T = TypeVar("T")
P = ParamSpec("P")


@overload
def cache(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def cache(*, cache_dir: str | Path, overwrite: bool = False) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def cache(
    func: Callable[P, T] | None = None,
    *,
    cache_dir: str | Path = "./cache",
    overwrite: bool = False,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """
    関数の実行結果をjoblibファイルにキャッシュするデコレータ。

    Args:
        func: キャッシュする関数（デコレータとして直接使用する場合）
        cache_dir: キャッシュファイルを保存するディレクトリ
        overwrite: Trueの場合、既存のキャッシュを上書きする

    Returns:
        デコレートされた関数または内部デコレータ

    Examples:
        # デコレータとして直接使用（デフォルトのcache_dir）
        @cache
        def my_function(x: int) -> int:
            return x * 2

        # パラメータ付きデコレータとして使用
        @cache(cache_dir="./my_cache", overwrite=False)
        def my_function(x: int) -> int:
            return x * 2
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        func_name = func.__name__
        cache_dir_path = Path(cache_dir)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # キャッシュディレクトリが存在しない場合は作成
            os.makedirs(cache_dir_path, exist_ok=True)

            # 引数のハッシュ値を計算（異なる引数で呼び出された場合に区別するため）
            arg_signature = _create_arg_signature(func, args, kwargs)
            hash_suffix = hashlib.md5(arg_signature.encode()).hexdigest()[:8]
            unique_cache_file = cache_dir_path / f"{func_name}_{hash_suffix}.joblib"

            # キャッシュが存在し、上書きしない場合はキャッシュを読み込む
            if not overwrite and unique_cache_file.exists():
                print(f"Cache hit: Loading from {unique_cache_file}")
                return cast(T, joblib.load(unique_cache_file))

            # 関数を実行
            result = func(*args, **kwargs)

            # 結果をキャッシュ
            joblib.dump(result, unique_cache_file)
            print(f"Cache miss: Saved to {unique_cache_file}")

            return result

        return wrapper

    # 直接デコレータとして使用された場合
    if func is not None:
        return decorator(func)

    # パラメータ付きデコレータとして使用された場合
    return decorator


def _create_arg_signature(func: Callable, args: tuple, kwargs: dict) -> str:
    """引数の値からシグネチャ文字列を作成"""
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    # 引数の値を文字列に変換
    args_str = []
    for name, value in bound_args.arguments.items():
        if hasattr(value, "__hash__") and value.__hash__ is not None:
            # ハッシュ可能なオブジェクト
            value_str = f"{name}:{str(value)}"
        elif isinstance(value, list):
            value_str = f"{name}:{str(value)}"
        elif isinstance(value, pl.DataFrame):
            value_str = f"{name}:DataFrame({value.head()})"
        else:
            print(f"Warning: Cannot hash {name}")
            continue
        args_str.append(value_str)

    return ",".join(args_str)
