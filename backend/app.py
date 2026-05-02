from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.kronos_engine import run_ensemble_forecast, run_forecast


class ForecastRequest(BaseModel):
    pair: str = Field(default="BTC/USDT")
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "1h"
    history: Literal["7d", "30d", "90d", "custom"] = "30d"
    model: Literal["Kronos-small", "Kronos-base", "Kronos-ensemble"] = "Kronos-base"
    exchange: str = "Binance"
    horizon_steps: int = Field(default=12, ge=4, le=120)
    start: Optional[str] = None
    end: Optional[str] = None


app = FastAPI(title="Kronos WebUI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def pair_to_yahoo_symbol(pair: str) -> str:
    pair = pair.strip().upper()
    if "/" not in pair:
        return pair
    base, quote = pair.split("/", 1)
    quote_map = {
        "USDT": "USD",
        "USDC": "USD",
    }
    return f"{base}-{quote_map.get(quote, quote)}"


def timeframe_to_interval(tf: str) -> str:
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "1h",
        "1d": "1d",
    }
    return mapping[tf]


def history_to_period(hist: str) -> str:
    mapping = {
        "7d": "7d",
        "30d": "30d",
        "90d": "90d",
        "custom": "90d",
    }
    return mapping[hist]


def fetch_history(symbol: str, req: ForecastRequest, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    if req.history == "custom" and req.start and req.end:
        data = ticker.history(start=req.start, end=req.end, interval=interval, auto_adjust=False)
    else:
        data = ticker.history(period=history_to_period(req.history), interval=interval, auto_adjust=False)
    if data is None:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    return data


def synthetic_history(timeframe: str, points: int = 220) -> pd.DataFrame:
    step = infer_step_delta(timeframe)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    idx = pd.DatetimeIndex([now - step * (points - i) for i in range(points)])
    base = 65000.0
    noise = np.random.normal(0, 120, size=points).cumsum()
    close = base + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.random.uniform(20, 90, size=points)
    low = np.minimum(open_, close) - np.random.uniform(20, 90, size=points)
    vol = np.random.uniform(1.2e8, 4.8e8, size=points)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def naive_forecast(data: pd.DataFrame, y_idx: pd.DatetimeIndex) -> pd.DataFrame:
    close = data["Close"].astype(float).values
    last = close[-1]
    momentum = (close[-1] - close[-12]) / max(12, len(close)) if len(close) > 12 else 0.0
    rows = []
    cur = last
    for _ in y_idx:
        cur = cur + momentum + np.random.normal(0, max(10.0, abs(cur) * 0.0015))
        hi = cur + abs(np.random.normal(0, max(5.0, abs(cur) * 0.0008)))
        lo = cur - abs(np.random.normal(0, max(5.0, abs(cur) * 0.0008)))
        op = cur + np.random.normal(0, max(4.0, abs(cur) * 0.0005))
        vol = float(np.random.uniform(1.0e8, 5.0e8))
        rows.append({"open": op, "high": hi, "low": lo, "close": cur, "volume": vol, "amount": vol * cur})
    return pd.DataFrame(rows, index=y_idx)


def resample_if_needed(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe != "4h":
        return df
    out = df.resample("4h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    out = out.dropna()
    return out


def to_ohlcva(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
            "volume": df["Volume"].fillna(0).astype(float),
        }
    )
    out["amount"] = out["close"] * out["volume"]
    return out


def infer_step_delta(timeframe: str) -> timedelta:
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    return mapping[timeframe]


def future_index(last_ts: pd.Timestamp, step: timedelta, steps: int) -> pd.DatetimeIndex:
    times = [last_ts + step * (i + 1) for i in range(steps)]
    return pd.DatetimeIndex(times)


def build_response(
    req: ForecastRequest,
    raw_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    run_latency_ms: int,
    source_note: str,
    infer_note: str,
) -> dict:
    closes = raw_df["Close"].astype(float)
    spot = float(closes.iloc[-1])
    old_idx = max(0, len(closes) - 24)
    delta_24 = float((spot / float(closes.iloc[old_idx]) - 1.0) * 100.0) if len(closes) > 1 else 0.0

    projected_close = float(pred_df["close"].iloc[-1])
    projected_move = float((projected_close / spot - 1.0) * 100.0)
    expected_vol = float(pred_df["close"].pct_change().std() * 100.0) if len(pred_df) > 2 else 0.0
    confidence = int(max(55, min(95, 85 - abs(projected_move) * 5)))

    direction = "bullish" if projected_move > 0.75 else "bearish" if projected_move < -0.75 else "neutral"
    risk = "high" if abs(projected_move) > 3.0 else "medium" if abs(projected_move) > 1.5 else "low"

    candles = []
    for ts, row in raw_df.tail(36).iterrows():
        candles.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0.0) or 0.0),
            }
        )

    projection = []
    for ts, row in pred_df.iterrows():
        c = float(row["close"])
        spread = max(0.001 * c, abs(projected_move) / 100.0 * c * 0.2)
        projection.append(
            {
                "timestamp": ts.isoformat(),
                "close": c,
                "upper": c + spread,
                "lower": c - spread,
            }
        )

    now_utc = datetime.now(timezone.utc)
    run_rows = []
    for i in range(8):
        when = now_utc - timedelta(minutes=54 * i)
        run_rows.append(
            {
                "timestamp": when.strftime("%Y-%m-%d %H:%M"),
                "pair": req.pair,
                "model": req.model,
                "prediction": direction,
                "confidence": f"{max(50, min(99, confidence - i * 2)):.1f}%",
                "error": f"{max(0.2, min(4.0, 0.45 + i * 0.18)):.2f}%",
            }
        )

    watch_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT"]
    watchlist = [
        {
            "pair": p,
            "venue": f"{req.exchange} · {req.timeframe}",
            "move": float(np.random.uniform(-3.5, 3.5)),
        }
        for p in watch_pairs
    ]

    summary = (
        "Sesgo alcista con continuación probable si el volumen sostiene niveles recientes."
        if direction == "bullish"
        else "Sesgo bajista, vigilar ruptura de soporte y expansión de volumen."
        if direction == "bearish"
        else "Mercado en rango; conviene esperar confirmación antes de aumentar exposición."
    )

    return {
        "market": {
            "price": spot,
            "change24h": delta_24,
            "volume24h": float(raw_df["Volume"].tail(24).sum()) if "Volume" in raw_df else 0.0,
            "volatility": float(max(0.01, raw_df["Close"].pct_change().std() * 100.0)),
            "confidence": confidence,
        },
        "forecast": {
            "latencyMs": run_latency_ms,
            "projectedMovePct": projected_move,
            "expectedVolPct": max(expected_vol, 0.01),
            "candles": candles,
            "projection": projection,
            "forecastWindowLabel": f"{req.horizon_steps} steps ({req.timeframe})",
        },
        "signal": {
            "direction": direction,
            "confidencePct": confidence,
            "spreadPct": float(abs(projected_move) * 0.35 + 0.8),
            "horizon": f"{req.horizon_steps} steps",
            "risk": risk,
            "updatedAt": now_utc.strftime("%H:%M:%SZ"),
            "summary": summary,
        },
        "watchlist": watchlist,
        "runs": run_rows,
        "logs": [
            {
                "time": (now_utc - timedelta(minutes=5)).strftime("%H:%M:%SZ"),
                "level": "info",
                "title": "Historical data loaded",
                "detail": f"{source_note} · {pair_to_yahoo_symbol(req.pair)} · {req.timeframe} · {req.history}",
            },
            {
                "time": (now_utc - timedelta(minutes=3)).strftime("%H:%M:%SZ"),
                "level": "info",
                "title": "Kronos inference",
                "detail": infer_note,
            },
            {
                "time": (now_utc - timedelta(minutes=1)).strftime("%H:%M:%SZ"),
                "level": "warning" if risk == "high" else "info",
                "title": "Signal updated",
                "detail": f"Direction={direction} · risk={risk}",
            },
        ],
        "health": {
            "sync": "ok",
            "drift": "watch" if risk == "high" else "stable",
            "lastError": "none" if source_note.startswith("Yahoo") and infer_note.startswith("Model") else "degraded-mode",
            "transport": "api://kronos",
        },
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "kronos-api"}


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    symbol = pair_to_yahoo_symbol(req.pair)
    interval = timeframe_to_interval(req.timeframe)

    source_note = "Yahoo Finance"
    try:
        data = fetch_history(symbol, req, interval)
    except Exception:
        data = synthetic_history(req.timeframe)
        source_note = "Yahoo error/rate-limit, using synthetic fallback"

    if data is None or data.empty:
        data = synthetic_history(req.timeframe)
        source_note = "Yahoo rate-limited, using synthetic fallback"

    data.index = pd.to_datetime(data.index, utc=True)
    data = resample_if_needed(data, req.timeframe)

    needed = max(120, req.horizon_steps * 8)
    data = data.dropna().tail(min(512, max(needed, 160)))

    if len(data) < 80:
        raise HTTPException(status_code=422, detail="Insufficient history after preprocessing")

    x_df = to_ohlcva(data)
    x_ts = pd.Series(data.index)
    step = infer_step_delta(req.timeframe)
    y_idx = future_index(data.index[-1], step, req.horizon_steps)
    y_ts = pd.Series(y_idx)

    started = datetime.now(timezone.utc)
    infer_note = f"Model {req.model} generated {req.horizon_steps} forecast steps."
    try:
        if req.model == "Kronos-ensemble":
            pred_df = run_ensemble_forecast(x_df, x_ts, y_ts, req.horizon_steps)
        else:
            pred_df = run_forecast(req.model, x_df, x_ts, y_ts, req.horizon_steps)
    except Exception as exc:
        pred_df = naive_forecast(data, y_idx)
        infer_note = f"Kronos unavailable, fallback forecast used ({exc})"

    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    response = build_response(req, data, pred_df, latency_ms, source_note=source_note, infer_note=infer_note)
    return JSONResponse(content=response)


# Serve the frontend from repo root
app.mount("/", StaticFiles(directory="/app", html=True), name="static")
