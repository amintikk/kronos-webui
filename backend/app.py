from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
from queue import Queue
import sqlite3
from threading import Thread
from typing import List, Literal, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.kronos_engine import run_ensemble_forecast, run_forecast


class ForecastRequest(BaseModel):
    pair: str = Field(default="BTC/USDT")
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "1h"
    history: Literal["7d", "15d", "30d", "90d", "custom"] = "30d"
    model: Literal["Kronos-small", "Kronos-base", "Kronos-ensemble"] = "Kronos-base"
    exchange: Literal["auto", "yahoo", "binance"] = "auto"
    horizon_steps: int = Field(default=12, ge=4, le=120)
    start: Optional[str] = None
    end: Optional[str] = None


class SearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str
    type: str


app = FastAPI(title="Kronos WebUI API", version="1.0.0")
DB_PATH = "/app/backend/runs.sqlite3"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              pair TEXT NOT NULL,
              timeframe TEXT NOT NULL,
              history TEXT NOT NULL,
              model TEXT NOT NULL,
              exchange_mode TEXT NOT NULL,
              horizon_steps INTEGER NOT NULL,
              source_note TEXT NOT NULL,
              spot_price REAL NOT NULL,
              projected_close REAL NOT NULL,
              projected_move_pct REAL NOT NULL,
              confidence INTEGER NOT NULL,
              risk TEXT NOT NULL,
              latency_ms INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_pair ON runs(pair)")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


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
        "15d": "15d",
        "30d": "30d",
        "90d": "90d",
        "custom": "90d",
    }
    return mapping[hist]


def binance_symbol_from_pair(pair: str) -> str:
    return pair.replace("/", "").upper()


def fetch_binance_history(pair: str, timeframe: str, history: str) -> pd.DataFrame:
    interval = timeframe
    limits = {"7d": 300, "15d": 500, "30d": 800, "90d": 1000, "custom": 1000}
    limit = limits.get(history, 800)
    params = urlencode({"symbol": binance_symbol_from_pair(pair), "interval": interval, "limit": limit})
    url = f"https://api.binance.com/api/v3/klines?{params}"
    with urlopen(url, timeout=15) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    out = []
    for r in rows:
        out.append(
            {
                "ts": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                "Open": float(r[1]),
                "High": float(r[2]),
                "Low": float(r[3]),
                "Close": float(r[4]),
                "Volume": float(r[5]),
            }
        )
    df = pd.DataFrame(out).set_index("ts").sort_index()
    return df


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


def search_yahoo_assets(query: str, limit: int = 10) -> List[SearchResult]:
    q = query.strip()
    if len(q) < 2:
        return []
    params = urlencode({"q": q, "quotesCount": limit, "newsCount": 0})
    url = f"https://query1.finance.yahoo.com/v1/finance/search?{params}"
    with urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    quotes = payload.get("quotes", []) if isinstance(payload, dict) else []
    results: List[SearchResult] = []
    for item in quotes:
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        results.append(
            SearchResult(
                symbol=symbol,
                name=str(item.get("shortname") or item.get("longname") or symbol),
                exchange=str(item.get("exchange") or ""),
                type=str(item.get("quoteType") or ""),
            )
        )
    return results


@lru_cache(maxsize=1)
def fetch_binance_exchange_info() -> list:
    with urlopen("https://api.binance.com/api/v3/exchangeInfo", timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("symbols", []) if isinstance(payload, dict) else []


def search_binance_assets(query: str, limit: int = 10) -> List[SearchResult]:
    q = query.strip().upper()
    if len(q) < 2:
        return []
    rows = fetch_binance_exchange_info()
    out: List[SearchResult] = []
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if q not in symbol:
            continue
        out.append(
            SearchResult(
                symbol=symbol,
                name="",
                exchange="BINANCE",
                type=str(row.get("status") or "SPOT"),
            )
        )
        if len(out) >= limit:
            break
    return out


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


def compute_probabilistic_bands(
    req: ForecastRequest,
    x_df: pd.DataFrame,
    x_ts: pd.Series,
    y_ts: pd.Series,
    steps: int,
    sample_runs: int,
    progress_cb=None,
) -> dict:
    close_paths = []
    volume_paths = []
    for _ in range(sample_runs):
        run_idx = len(close_paths) + 1
        if progress_cb:
            progress_cb("inference", f"Running stochastic sample {run_idx}/{sample_runs}", 55 + int((run_idx - 1) * 30 / max(1, sample_runs)))
        if req.model == "Kronos-ensemble":
            sample_df = run_ensemble_forecast(x_df, x_ts, y_ts, steps)
        else:
            sample_df = run_forecast(req.model, x_df, x_ts, y_ts, steps)
        close_paths.append(sample_df["close"].astype(float).to_numpy())
        volume_paths.append(sample_df["volume"].astype(float).to_numpy())

    close_arr = np.asarray(close_paths)  # [runs, steps]
    vol_arr = np.asarray(volume_paths)   # [runs, steps]
    return {
        "close_p10": np.quantile(close_arr, 0.10, axis=0),
        "close_p50": np.quantile(close_arr, 0.50, axis=0),
        "close_p90": np.quantile(close_arr, 0.90, axis=0),
        "close_mean": np.mean(close_arr, axis=0),
        "volume_mean": np.mean(vol_arr, axis=0),
        "sample_runs": sample_runs,
    }


def build_response(
    req: ForecastRequest,
    raw_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    run_latency_ms: int,
    source_note: str,
    infer_note: str,
    prob_bands: Optional[dict] = None,
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

    days_by_history = {"7d": 7, "15d": 15, "30d": 30, "90d": 90, "custom": 90}
    step = infer_step_delta(req.timeframe)
    target_points = int((days_by_history.get(req.history, 30) * 24 * 3600) / max(1.0, step.total_seconds()))
    target_points = max(80, min(2000, target_points))

    candles = []
    for ts, row in raw_df.tail(target_points).iterrows():
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
    close_p10 = prob_bands.get("close_p10") if prob_bands else None
    close_p50 = prob_bands.get("close_p50") if prob_bands else None
    close_p90 = prob_bands.get("close_p90") if prob_bands else None
    vol_mean = prob_bands.get("volume_mean") if prob_bands else None

    for idx, (ts, row) in enumerate(pred_df.iterrows()):
        c_default = float(row["close"])
        c = float(close_p50[idx]) if close_p50 is not None else c_default
        low = float(close_p10[idx]) if close_p10 is not None else max(0.0, c_default * 0.999)
        high = float(close_p90[idx]) if close_p90 is not None else c_default * 1.001
        if high < low:
            low, high = high, low
        v = float(row.get("volume", 0.0) or 0.0)
        if vol_mean is not None:
            v = float(vol_mean[idx])
        projection.append(
            {
                "timestamp": ts.isoformat(),
                "close": c,
                "upper": high,
                "lower": low,
                "volume": v,
            }
        )

    now_utc = datetime.now(timezone.utc)

    summary = (
        "Sesgo alcista con continuación probable si el volumen sostiene niveles recientes."
        if direction == "bullish"
        else "Sesgo bajista, vigilar ruptura de soporte y expansión de volumen."
        if direction == "bearish"
        else "Mercado en rango; conviene esperar confirmación antes de aumentar exposición."
    )

    horizon_24h_steps = max(1, int(round(timedelta(hours=24).total_seconds() / max(1.0, step.total_seconds()))))
    horizon_24h_steps = min(horizon_24h_steps, len(pred_df))

    pred_close_series = pred_df["close"].astype(float)
    pred_next_24h = pred_close_series.head(horizon_24h_steps)
    pred_min = float(pred_close_series.min()) if len(pred_close_series) else None
    pred_max = float(pred_close_series.max()) if len(pred_close_series) else None
    pred_mean = float(pred_close_series.mean()) if len(pred_close_series) else None

    upside_prob_24h = None
    if len(pred_next_24h):
        upside_prob_24h = float((pred_next_24h > spot).mean() * 100.0)

    hist_ret_24 = raw_df["Close"].astype(float).pct_change().dropna().tail(horizon_24h_steps)
    hist_vol_24 = float(hist_ret_24.std() * 100.0) if len(hist_ret_24) > 1 else None
    pred_ret_24 = pred_next_24h.pct_change().dropna()
    pred_vol_24 = float(pred_ret_24.std() * 100.0) if len(pred_ret_24) > 1 else None
    vol_amp_24h = float(pred_vol_24 / hist_vol_24) if (hist_vol_24 and hist_vol_24 > 0 and pred_vol_24 is not None) else None

    hist_vol_mean = float(raw_df["Volume"].tail(horizon_24h_steps).mean()) if "Volume" in raw_df and len(raw_df) else None
    pred_vol_mean = float(pred_df["volume"].astype(float).head(horizon_24h_steps).mean()) if "volume" in pred_df else None
    exp_return_24h = float((pred_next_24h.iloc[-1] / spot - 1.0) * 100.0) if len(pred_next_24h) else None
    max_drawdown_24h = None
    if len(pred_next_24h):
        rolling_peak = pred_next_24h.cummax()
        drawdown = (pred_next_24h / rolling_peak - 1.0) * 100.0
        max_drawdown_24h = float(drawdown.min())

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
        "watchlist": [],
        "runs": [],
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
            "lastError": "none",
            "transport": "api://kronos",
        },
        "analytics": {
            "forecast_range_min": pred_min,
            "forecast_range_max": pred_max,
            "mean_forecast": pred_mean,
            "historical_price": spot,
            "historical_volume_mean": hist_vol_mean,
            "mean_forecasted_volume": pred_vol_mean,
            "upside_probability_next_24h": upside_prob_24h,
            "volatility_amplification_next_24h": vol_amp_24h,
            "expected_return_next_24h": exp_return_24h,
            "max_drawdown_next_24h": max_drawdown_24h,
            "historical_volatility_next_24h_window": hist_vol_24,
            "forecast_volatility_next_24h_window": pred_vol_24,
            "window_steps_24h": horizon_24h_steps,
            "probabilistic_samples": int(prob_bands.get("sample_runs", 0)) if prob_bands else 0,
        },
    }


def save_run(
    req: ForecastRequest,
    response: dict,
    source_note: str,
) -> None:
    market = response.get("market", {})
    signal = response.get("signal", {})
    forecast = response.get("forecast", {})
    projection = forecast.get("projection", [])
    projected_close = projection[-1]["close"] if projection else market.get("price", 0)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO runs (
              created_at, pair, timeframe, history, model, exchange_mode, horizon_steps,
              source_note, spot_price, projected_close, projected_move_pct, confidence, risk, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                req.pair,
                req.timeframe,
                req.history,
                req.model,
                req.exchange,
                req.horizon_steps,
                source_note,
                float(market.get("price") or 0.0),
                float(projected_close or 0.0),
                float(forecast.get("projectedMovePct") or 0.0),
                int(market.get("confidence") or 0),
                str(signal.get("risk") or "-"),
                int(forecast.get("latencyMs") or 0),
            ),
        )


def fetch_recent_runs(limit: int = 20) -> List[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT created_at, pair, model, projected_move_pct, confidence, risk
            FROM runs
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = []
    for row in rows:
        dt = datetime.fromisoformat(row["created_at"])
        out.append(
            {
                "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
                "pair": row["pair"],
                "model": row["model"],
                "prediction": "bullish" if row["projected_move_pct"] > 0 else "bearish" if row["projected_move_pct"] < 0 else "neutral",
                "confidence": f'{float(row["confidence"]):.1f}%',
                "error": f'{abs(float(row["projected_move_pct"])):.2f}%',
                "risk": row["risk"],
            }
        )
    return out


def compute_forecast_response(req: ForecastRequest, progress_cb=None) -> dict:
    def emit(stage: str, message: str, percent: int):
        if progress_cb:
            progress_cb(stage, message, percent)

    emit("start", "Validating request", 5)
    symbol = pair_to_yahoo_symbol(req.pair)
    interval = timeframe_to_interval(req.timeframe)

    data = pd.DataFrame()
    source_note = ""
    emit("data", f"Loading market history ({req.exchange})", 10)
    if req.exchange == "yahoo":
        try:
            data = fetch_history(symbol, req, interval)
            source_note = "Yahoo Finance"
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yahoo fetch failed: {exc}")
    elif req.exchange == "binance":
        try:
            data = fetch_binance_history(req.pair, req.timeframe, req.history)
            source_note = "Binance public klines"
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Binance fetch failed: {exc}")
    else:
        try:
            data = fetch_history(symbol, req, interval)
            source_note = "Yahoo Finance"
        except Exception:
            data = pd.DataFrame()
        if data is None or data.empty:
            try:
                data = fetch_binance_history(req.pair, req.timeframe, req.history)
                source_note = "Binance public klines (auto-fallback)"
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to load history from Yahoo and Binance fallback: {exc}",
                )

    if data is None or data.empty:
        raise HTTPException(status_code=502, detail="No market history returned from upstream providers")
    emit("data", f"History loaded from {source_note} ({len(data)} rows)", 25)

    data.index = pd.to_datetime(data.index, utc=True)
    data = resample_if_needed(data, req.timeframe)

    needed = max(120, req.horizon_steps * 8)
    data = data.dropna()

    if len(data) < 80:
        raise HTTPException(status_code=422, detail="Insufficient history after preprocessing")
    emit("preprocess", f"Preprocessing complete ({len(data)} clean rows)", 40)

    data_for_model = data.tail(min(512, max(needed, 160)))
    x_df = to_ohlcva(data_for_model)
    x_ts = pd.Series(data_for_model.index)
    step = infer_step_delta(req.timeframe)
    y_idx = future_index(data_for_model.index[-1], step, req.horizon_steps)
    y_ts = pd.Series(y_idx)

    sample_runs = 7 if req.model != "Kronos-ensemble" else 5
    started = datetime.now(timezone.utc)
    emit("inference", f"Starting Kronos probabilistic inference ({sample_runs} runs)", 50)
    try:
        prob_bands = compute_probabilistic_bands(
            req=req,
            x_df=x_df,
            x_ts=x_ts,
            y_ts=y_ts,
            steps=req.horizon_steps,
            sample_runs=sample_runs,
            progress_cb=progress_cb,
        )
        if req.model == "Kronos-ensemble":
            pred_df = run_ensemble_forecast(x_df, x_ts, y_ts, req.horizon_steps)
        else:
            pred_df = run_forecast(req.model, x_df, x_ts, y_ts, req.horizon_steps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kronos inference failed: {exc}")
    emit("inference", "Inference completed", 90)

    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    response = build_response(
        req,
        data,
        pred_df,
        latency_ms,
        source_note=source_note,
        infer_note=f"Model {req.model} generated {req.horizon_steps} forecast steps with {sample_runs} stochastic runs (p10/p50/p90).",
        prob_bands=prob_bands,
    )
    save_run(req, response, source_note=source_note)
    response["runs"] = fetch_recent_runs(limit=20)
    emit("done", "Response assembled", 100)
    return response


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "kronos-api"}


@app.get("/api/search")
def search_assets(q: str) -> dict:
    try:
        items = search_yahoo_assets(q, limit=10)
        if not items:
            items = search_binance_assets(q, limit=10)
    except Exception:
        try:
            items = search_binance_assets(q, limit=10)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Asset search failed (Yahoo + Binance): {exc}")
    return {"items": [item.model_dump() for item in items]}


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    response = compute_forecast_response(req=req, progress_cb=None)
    return JSONResponse(content=response)


@app.post("/api/forecast/stream")
def forecast_stream(req: ForecastRequest):
    q: Queue = Queue()

    def progress_cb(stage: str, message: str, percent: int):
        q.put({"type": "progress", "stage": stage, "message": message, "percent": percent})

    def worker():
        try:
            result = compute_forecast_response(req=req, progress_cb=progress_cb)
            q.put({"type": "result", "payload": result})
        except HTTPException as exc:
            q.put({"type": "error", "status": exc.status_code, "message": str(exc.detail)})
        except Exception as exc:
            q.put({"type": "error", "status": 500, "message": str(exc)})
        finally:
            q.put({"type": "done"})

    Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            yield json.dumps(item) + "\n"
            if item.get("type") == "done":
                break

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# Serve the frontend from repo root
app.mount("/", StaticFiles(directory="/app", html=True), name="static")
