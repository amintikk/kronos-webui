from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from model import Kronos, KronosPredictor, KronosTokenizer

MODEL_SPECS: Dict[str, Tuple[str, str]] = {
    "Kronos-mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k"),
    "Kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
    "Kronos-base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base"),
}


@dataclass
class LoadedModel:
    predictor: KronosPredictor
    model_name: str


@lru_cache(maxsize=3)
def load_predictor(model_name: str) -> LoadedModel:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported Kronos model: {model_name}")

    model_repo, tokenizer_repo = MODEL_SPECS[model_name]
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo)
    model = Kronos.from_pretrained(model_repo)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    return LoadedModel(predictor=predictor, model_name=model_name)


def run_forecast(
    model_name: str,
    history_df: pd.DataFrame,
    x_ts: pd.Series,
    y_ts: pd.Series,
    pred_len: int,
    sample_count: int = 1,
) -> pd.DataFrame:
    loaded = load_predictor(model_name)
    close_df, volume_df = loaded.predictor.predict(
        df=history_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0,
        top_p=0.95,
        sample_count=sample_count,
        verbose=False,
    )
    return pd.DataFrame(
        {
            "close": close_df.mean(axis=1).astype(float),
            "volume": volume_df.mean(axis=1).astype(float),
        },
        index=close_df.index,
    )


def run_forecast_paths(
    model_name: str,
    history_df: pd.DataFrame,
    x_ts: pd.Series,
    y_ts: pd.Series,
    pred_len: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    loaded = load_predictor(model_name)
    close_df, volume_df = loaded.predictor.predict(
        df=history_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0,
        top_p=0.95,
        sample_count=sample_count,
        verbose=False,
    )
    return close_df.to_numpy(dtype=float).T, volume_df.to_numpy(dtype=float).T


def run_ensemble_forecast(
    history_df: pd.DataFrame,
    x_ts: pd.Series,
    y_ts: pd.Series,
    pred_len: int,
) -> pd.DataFrame:
    pred_small = run_forecast("Kronos-small", history_df, x_ts, y_ts, pred_len)
    pred_base = run_forecast("Kronos-base", history_df, x_ts, y_ts, pred_len)
    ensemble = (pred_small + pred_base) / 2.0
    return ensemble
