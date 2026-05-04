from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import os
from queue import Queue
import sqlite3
from threading import Thread
from threading import Lock
from typing import List, Literal, Optional
from uuid import uuid4
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.kronos_engine import run_ensemble_forecast, run_forecast, run_forecast_paths


class ForecastRequest(BaseModel):
    pair: str = Field(default="BTCUSDT")
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "1h"
    history: Literal["1d", "2d", "3d", "7d", "15d", "30d", "60d", "90d", "custom"] = "15d"
    model: Literal["Kronos-mini", "Kronos-small", "Kronos-base", "Kronos-ensemble"] = "Kronos-base"
    exchange: Literal["auto", "yahoo", "binance"] = "binance"
    horizon_steps: int = Field(default=12, ge=4, le=120)
    sample_runs: int = Field(default=7, ge=1, le=40)
    start: Optional[str] = None
    end: Optional[str] = None


class SearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str
    type: str


class SaveRunRequest(BaseModel):
    request: ForecastRequest
    response: dict
    source_note: str = "Manual save"


app = FastAPI(title="Kronos WebUI API", version="1.0.0")
DB_PATH = os.getenv("DB_PATH", "/data/runs.sqlite3")
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()
MAX_JOBS = 100

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
              latency_ms INTEGER NOT NULL,
              response_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_pair ON runs(pair)")
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()]
        if "response_json" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN response_json TEXT NOT NULL DEFAULT '{}'")
        if "actual_close" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN actual_close REAL")
        if "backtest_error_pct" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN backtest_error_pct REAL")
        if "backtest_abs_error_pct" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN backtest_abs_error_pct REAL")
        if "evaluated_at" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN evaluated_at TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_jobs (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              status TEXT NOT NULL,
              request_json TEXT NOT NULL,
              events_json TEXT NOT NULL DEFAULT '[]',
              result_json TEXT,
              error_text TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_jobs_created_at ON forecast_jobs(created_at DESC)")


def _persist_job(job: dict) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO forecast_jobs (id, created_at, updated_at, status, request_json, events_json, result_json, error_text, cancel_requested)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              updated_at=excluded.updated_at,
              status=excluded.status,
              request_json=excluded.request_json,
              events_json=excluded.events_json,
              result_json=excluded.result_json,
              error_text=excluded.error_text,
              cancel_requested=excluded.cancel_requested
            """,
            (
                job["id"],
                job["created_at"],
                job["updated_at"],
                job["status"],
                json.dumps(job.get("request") or {}),
                json.dumps(job.get("events") or []),
                json.dumps(job.get("result")) if job.get("result") is not None else None,
                job.get("error"),
                int(bool(job.get("cancel_requested"))),
            ),
        )


def _new_job(req: ForecastRequest) -> str:
    job_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "request": req.model_dump(),
        "events": [],
        "result": None,
        "error": None,
        "cancel_requested": False,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        if len(JOBS) > MAX_JOBS:
            ordered = sorted(JOBS.items(), key=lambda kv: kv[1].get("created_at", ""))
            for old_id, _ in ordered[: max(0, len(JOBS) - MAX_JOBS)]:
                JOBS.pop(old_id, None)
    _persist_job(job)
    return job_id


def _append_job_event(job_id: str, stage: str, message: str, percent: int):
    now = datetime.now(timezone.utc).isoformat()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        events = job.setdefault("events", [])
        events.append(
            {
                "idx": len(events),
                "ts": now,
                "stage": stage,
                "message": message,
                "percent": int(percent),
            }
        )
        job["updated_at"] = now
        _persist_job(job)


def _set_job_state(job_id: str, *, status: str, result: Optional[dict] = None, error: Optional[str] = None):
    now = datetime.now(timezone.utc).isoformat()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = status
        job["updated_at"] = now
        job["result"] = result
        job["error"] = error
        _persist_job(job)


def _get_job_row(job_id: str):
    with get_db() as conn:
        return conn.execute(
            """
            SELECT id, created_at, updated_at, status, request_json, events_json, result_json, error_text, cancel_requested
            FROM forecast_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, updated_at, status, request_json, events_json, result_json, error_text, cancel_requested FROM forecast_jobs ORDER BY datetime(created_at) DESC LIMIT ?",
            (MAX_JOBS,),
        ).fetchall()
    with JOBS_LOCK:
        JOBS.clear()
        for row in rows:
            try:
                req_json = json.loads(row["request_json"] or "{}")
            except Exception:
                req_json = {}
            try:
                events = json.loads(row["events_json"] or "[]")
            except Exception:
                events = []
            try:
                result = json.loads(row["result_json"]) if row["result_json"] else None
            except Exception:
                result = None
            status = str(row["status"] or "queued")
            error_text = row["error_text"]
            if status in ("queued", "running"):
                status = "failed"
                error_text = "Server restarted while job was running"
            JOBS[str(row["id"])] = {
                "id": str(row["id"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "status": status,
                "request": req_json,
                "events": events,
                "result": result,
                "error": error_text,
                "cancel_requested": bool(int(row["cancel_requested"] or 0)),
            }
        for job in JOBS.values():
            _persist_job(job)


def pair_to_yahoo_symbol(pair: str) -> str:
    pair = pair.strip().upper()
    base = ""
    quote = ""
    if "/" in pair:
        base, quote = pair.split("/", 1)
    else:
        known_quotes = ["USDT", "USDC", "USD", "BTC", "ETH", "BNB", "EUR", "TRY"]
        for q in known_quotes:
            if pair.endswith(q) and len(pair) > len(q):
                base = pair[: -len(q)]
                quote = q
                break
        if not base or not quote:
            return pair
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
        "1d": "1d",
        "2d": "2d",
        "3d": "3d",
        "7d": "7d",
        "15d": "15d",
        "30d": "30d",
        "60d": "60d",
        "90d": "90d",
        "custom": "90d",
    }
    return mapping[hist]


def binance_symbol_from_pair(pair: str) -> str:
    return pair.replace("/", "").upper()


def fetch_binance_history(
    pair: str,
    timeframe: str,
    history: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    interval = timeframe
    symbol = binance_symbol_from_pair(pair)
    step_seconds = max(1, int(infer_step_delta(timeframe).total_seconds()))

    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        step_ms = step_seconds * 1000

        collected = []
        seen_open_times = set()
        while start_ms < end_ms:
            query = {
                "symbol": symbol,
                "interval": interval,
                "limit": 1000,
                "startTime": start_ms,
                "endTime": end_ms,
            }
            params = urlencode(query)
            url = f"https://api.binance.com/api/v3/klines?{params}"
            with urlopen(url, timeout=20) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            if not isinstance(rows, list) or not rows:
                break

            for r in rows:
                open_ms = int(r[0])
                if open_ms in seen_open_times:
                    continue
                seen_open_times.add(open_ms)
                collected.append(
                    {
                        "ts": pd.to_datetime(open_ms, unit="ms", utc=True),
                        "Open": float(r[1]),
                        "High": float(r[2]),
                        "Low": float(r[3]),
                        "Close": float(r[4]),
                        "Volume": float(r[5]),
                        "QuoteVolume": float(r[7]),
                    }
                )

            last_open_ms = int(rows[-1][0])
            next_start = last_open_ms + step_ms
            if next_start <= start_ms:
                break
            start_ms = next_start

            if len(rows) < 1000:
                break

        if not collected:
            return pd.DataFrame()
        df = pd.DataFrame(collected).set_index("ts").sort_index()
        return df

    days_by_history = {"1d": 1, "2d": 2, "3d": 3, "7d": 7, "15d": 15, "30d": 30, "60d": 60, "90d": 90, "custom": 90}
    target_days = days_by_history.get(history, 30)
    target_bars = int((target_days * 24 * 3600) / step_seconds) + 32
    target_bars = max(300, min(8000, target_bars))

    collected = []
    seen_open_times = set()
    end_time_ms: Optional[int] = None

    while len(collected) < target_bars:
        per_call = min(1000, target_bars - len(collected))
        query = {"symbol": symbol, "interval": interval, "limit": per_call}
        if end_time_ms is not None:
            query["endTime"] = end_time_ms
        params = urlencode(query)
        url = f"https://api.binance.com/api/v3/klines?{params}"
        with urlopen(url, timeout=20) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            break

        chunk = []
        for r in rows:
            open_ms = int(r[0])
            if open_ms in seen_open_times:
                continue
            seen_open_times.add(open_ms)
            chunk.append(
                {
                    "ts": pd.to_datetime(open_ms, unit="ms", utc=True),
                    "Open": float(r[1]),
                    "High": float(r[2]),
                    "Low": float(r[3]),
                    "Close": float(r[4]),
                    "Volume": float(r[5]),
                    "QuoteVolume": float(r[7]),
                }
            )
        if not chunk:
            break

        collected = chunk + collected
        oldest_open_ms = int(rows[0][0])
        next_end = oldest_open_ms - 1
        if end_time_ms is not None and next_end >= end_time_ms:
            break
        end_time_ms = next_end

        if len(rows) < per_call:
            break

    if not collected:
        return pd.DataFrame()
    df = pd.DataFrame(collected).set_index("ts").sort_index()
    return df.tail(target_bars)


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
    if "QuoteVolume" in df.columns:
        out["amount"] = df["QuoteVolume"].fillna(0).astype(float)
    else:
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
    def emit_progress(done: int, total: int):
        if not progress_cb:
            return
        pct = 55 + int((max(0, min(done, total))) * 29 / max(1, total))
        progress_cb("inference", f"Running stochastic batch {done}/{total}", pct)

    batch_size = 1
    if req.model == "Kronos-ensemble":
        close_small_parts = []
        vol_small_parts = []
        close_base_parts = []
        vol_base_parts = []
        done = 0
        while done < sample_runs:
            take = min(batch_size, sample_runs - done)
            emit_progress(done + 1, sample_runs)
            c_s, v_s = run_forecast_paths("Kronos-small", x_df, x_ts, y_ts, steps, take)
            c_b, v_b = run_forecast_paths("Kronos-base", x_df, x_ts, y_ts, steps, take)
            close_small_parts.append(c_s)
            vol_small_parts.append(v_s)
            close_base_parts.append(c_b)
            vol_base_parts.append(v_b)
            done += take
            emit_progress(done, sample_runs)
        close_small = np.concatenate(close_small_parts, axis=0)
        vol_small = np.concatenate(vol_small_parts, axis=0)
        close_base = np.concatenate(close_base_parts, axis=0)
        vol_base = np.concatenate(vol_base_parts, axis=0)
        close_arr = (close_small + close_base) / 2.0
        vol_arr = (vol_small + vol_base) / 2.0
    else:
        close_parts = []
        vol_parts = []
        done = 0
        while done < sample_runs:
            take = min(batch_size, sample_runs - done)
            emit_progress(done + 1, sample_runs)
            c, v = run_forecast_paths(req.model, x_df, x_ts, y_ts, steps, take)
            close_parts.append(c)
            vol_parts.append(v)
            done += take
            emit_progress(done, sample_runs)
        close_arr = np.concatenate(close_parts, axis=0)
        vol_arr = np.concatenate(vol_parts, axis=0)

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

    days_by_history = {"1d": 1, "2d": 2, "3d": 3, "7d": 7, "15d": 15, "30d": 30, "60d": 60, "90d": 90, "custom": 90}
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

    now_es = datetime.now(ZoneInfo("Europe/Madrid"))

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
            "updatedAt": now_es.strftime("%H:%M:%S"),
            "summary": summary,
        },
        "watchlist": [],
        "runs": [],
        "logs": [
            {
                "time": (now_es - timedelta(minutes=5)).strftime("%H:%M:%S"),
                "level": "info",
                "title": "Historical data loaded",
                "detail": f"{source_note} · {pair_to_yahoo_symbol(req.pair)} · {req.timeframe} · {req.history}",
            },
            {
                "time": (now_es - timedelta(minutes=3)).strftime("%H:%M:%S"),
                "level": "info",
                "title": "Kronos inference",
                "detail": infer_note,
            },
            {
                "time": (now_es - timedelta(minutes=1)).strftime("%H:%M:%S"),
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
              source_note, spot_price, projected_close, projected_move_pct, confidence, risk, latency_ms, response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(response),
            ),
        )


def fetch_realized_close_for_run(row: sqlite3.Row, target_time: datetime) -> Optional[float]:
    pair = str(row["pair"])
    timeframe = str(row["timeframe"])
    exchange_mode = str(row["exchange_mode"] or "binance")
    step = infer_step_delta(timeframe)
    start = target_time - step
    end = target_time + step * 2

    try:
        if exchange_mode == "binance":
            df = fetch_binance_history(pair, timeframe, "custom", start_time=start, end_time=end)
        elif exchange_mode == "yahoo":
            req = ForecastRequest(
                pair=pair, timeframe=timeframe, history="custom", model="Kronos-base", exchange="yahoo",
                horizon_steps=12, start=start.isoformat(), end=end.isoformat()
            )
            df = fetch_history(pair_to_yahoo_symbol(pair), req, timeframe_to_interval(timeframe))
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                df = resample_if_needed(df, timeframe)
        else:
            df = fetch_binance_history(pair, timeframe, "custom", start_time=start, end_time=end)
    except Exception:
        return None

    if df is None or df.empty:
        return None
    df = df.sort_index()
    target_ts = pd.Timestamp(target_time)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("UTC")
    else:
        target_ts = target_ts.tz_convert("UTC")
    deltas = (df.index - target_ts).asi8
    closest_pos = int(np.abs(deltas).argmin())
    return float(df.iloc[closest_pos]["Close"])


def evaluate_due_runs(max_rows: int = 100) -> None:
    now_utc = datetime.now(timezone.utc)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, pair, timeframe, exchange_mode, horizon_steps, projected_close, actual_close
            FROM runs
            WHERE actual_close IS NULL
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (max_rows,),
        ).fetchall()

        for row in rows:
            try:
                created_at = datetime.fromisoformat(str(row["created_at"]))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            step = infer_step_delta(str(row["timeframe"]))
            horizon_steps = int(row["horizon_steps"] or 0)
            if horizon_steps <= 0:
                continue
            target_time = created_at + step * horizon_steps
            if now_utc < target_time:
                continue

            actual_close = fetch_realized_close_for_run(row, target_time)
            if actual_close is None:
                continue

            projected_close = float(row["projected_close"] or 0.0)
            if projected_close == 0:
                err_pct = None
                abs_err_pct = None
            else:
                err_pct = ((actual_close - projected_close) / abs(projected_close)) * 100.0
                abs_err_pct = abs(err_pct)

            conn.execute(
                """
                UPDATE runs
                SET actual_close = ?, backtest_error_pct = ?, backtest_abs_error_pct = ?, evaluated_at = ?
                WHERE id = ?
                """,
                (
                    actual_close,
                    float(err_pct) if err_pct is not None else None,
                    float(abs_err_pct) if abs_err_pct is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                    int(row["id"]),
                ),
            )


def fetch_recent_runs(limit: int = 20) -> List[dict]:
    evaluate_due_runs(max_rows=200)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, pair, model, projected_move_pct, confidence, risk, backtest_abs_error_pct
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
                "id": int(row["id"]),
                "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
                "pair": row["pair"],
                "model": row["model"],
                "prediction": "bullish" if row["projected_move_pct"] > 0 else "bearish" if row["projected_move_pct"] < 0 else "neutral",
                "confidence": f'{float(row["confidence"]):.1f}%',
                "error": (
                    f'{float(row["backtest_abs_error_pct"]):.2f}%'
                    if row["backtest_abs_error_pct"] is not None
                    else f'{abs(float(row["projected_move_pct"])):.2f}%'
                ),
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

    # Use only closed candles: drop the last bar only if it is still in progress.
    data_for_model = data.copy()
    if len(data_for_model) > 81:
        step = infer_step_delta(req.timeframe)
        last_ts = pd.Timestamp(data_for_model.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        last_bar_closed = now_utc >= (last_ts + step)
        if not last_bar_closed:
            data_for_model = data_for_model.iloc[:-1]

    # Volatility-adaptive context: larger memory in high-vol regimes.
    recent_returns = data_for_model["Close"].astype(float).pct_change().dropna().tail(96)
    realized_vol_pct = float(recent_returns.std() * 100.0) if len(recent_returns) > 2 else 0.0
    if realized_vol_pct >= 1.8:
        regime = "high-vol"
        adaptive_context = 512
    elif realized_vol_pct >= 0.9:
        regime = "mid-vol"
        adaptive_context = 384
    else:
        regime = "low-vol"
        adaptive_context = 256
    context_len = min(512, max(needed, adaptive_context))
    emit("preprocess", f"Adaptive context={context_len} ({regime}, vol={realized_vol_pct:.2f}%)", 45)

    data_for_model = data_for_model.tail(context_len)
    x_df = to_ohlcva(data_for_model)
    x_ts = pd.Series(data_for_model.index)
    step = infer_step_delta(req.timeframe)
    y_idx = future_index(data_for_model.index[-1], step, req.horizon_steps)
    y_ts = pd.Series(y_idx)

    sample_runs = int(req.sample_runs or (7 if req.model != "Kronos-ensemble" else 5))
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

    # Quality gate: de-risk ambiguous forecasts with wide uncertainty.
    try:
        spot = float(response.get("market", {}).get("price") or 0.0)
        upside_24 = response.get("analytics", {}).get("upside_probability_next_24h")
        p10 = prob_bands.get("close_p10") if prob_bands else None
        p90 = prob_bands.get("close_p90") if prob_bands else None
        last_band_pct = None
        if p10 is not None and p90 is not None and len(p10) and len(p90) and spot > 0:
            last_band_pct = float((float(p90[-1]) - float(p10[-1])) / spot * 100.0)

        ambiguous_upside = isinstance(upside_24, (int, float)) and 45.0 <= float(upside_24) <= 55.0
        wide_band = isinstance(last_band_pct, float) and last_band_pct >= 3.0
        if ambiguous_upside and wide_band:
            response["signal"]["direction"] = "neutral"
            response["signal"]["risk"] = "high"
            response["signal"]["confidencePct"] = min(int(response["signal"].get("confidencePct", 60)), 60)
            response["signal"]["summary"] = (
                "Señal neutral por quality gate: incertidumbre alta y probabilidad alcista cercana a 50%."
            )
            response["logs"] = [
                {
                    "time": datetime.now(ZoneInfo("Europe/Madrid")).strftime("%H:%M:%S"),
                    "level": "warning",
                    "title": "Quality gate active",
                    "detail": f"Ambiguous upside ({float(upside_24):.1f}%) with wide forecast band ({float(last_band_pct):.2f}%).",
                }
            ] + response.get("logs", [])
    except Exception:
        pass

    response["runs"] = fetch_recent_runs(limit=20)
    response["request_context"] = req.model_dump()
    response["source_note"] = source_note
    emit("done", "Response assembled", 100)
    return response


def load_history_for_run_context(
    pair: str,
    timeframe: str,
    history: str,
    exchange_mode: str,
    min_days: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> tuple[pd.DataFrame, str]:
    symbol = pair_to_yahoo_symbol(pair)
    interval = timeframe_to_interval(timeframe)
    replay_history = history
    if min_days is not None:
        d = max(1, int(min_days))
        if d <= 7:
            replay_history = "7d"
        elif d <= 15:
            replay_history = "15d"
        elif d <= 30:
            replay_history = "30d"
        else:
            replay_history = "90d"
    req = ForecastRequest(
        pair=pair,
        timeframe=timeframe,
        history=replay_history,
        model="Kronos-base",
        exchange=exchange_mode if exchange_mode in ("auto", "yahoo", "binance") else "binance",
        horizon_steps=12,
    )
    data = pd.DataFrame()
    source_note = ""
    if req.exchange == "yahoo":
        if min_days is not None:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{max(7, int(min_days))}d", interval=interval, auto_adjust=False)
        else:
            data = fetch_history(symbol, req, interval)
        source_note = "Yahoo Finance"
    elif req.exchange == "binance":
        data = fetch_binance_history(req.pair, req.timeframe, req.history, start_time=start_time, end_time=end_time)
        if (data is None or data.empty) and min_days is not None:
            # Fallback for very recent runs/timeframes where exact range returns no bar yet.
            data = fetch_binance_history(req.pair, req.timeframe, req.history)
        source_note = "Binance public klines"
    else:
        try:
            data = fetch_history(symbol, req, interval)
            source_note = "Yahoo Finance"
        except Exception:
            data = pd.DataFrame()
        if data is None or data.empty:
            data = fetch_binance_history(req.pair, req.timeframe, req.history, start_time=start_time, end_time=end_time)
            source_note = "Binance public klines (auto-fallback)"
    if data is None or data.empty:
        raise HTTPException(status_code=502, detail="Could not load updated history for replay")
    data.index = pd.to_datetime(data.index, utc=True)
    data = resample_if_needed(data, timeframe).dropna()
    return data, source_note


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


@app.post("/api/forecast/jobs")
def forecast_job_start(req: ForecastRequest):
    job_id = _new_job(req)
    _set_job_state(job_id, status="running")
    _append_job_event(job_id, "start", "Job accepted", 1)

    def progress_cb(stage: str, message: str, percent: int):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job and job.get("cancel_requested"):
                raise RuntimeError("Job cancelled by user")
        _append_job_event(job_id, stage, message, percent)

    def worker():
        try:
            result = compute_forecast_response(req=req, progress_cb=progress_cb)
            _append_job_event(job_id, "done", "Response assembled", 100)
            _set_job_state(job_id, status="completed", result=result, error=None)
        except HTTPException as exc:
            _append_job_event(job_id, "error", str(exc.detail), 100)
            _set_job_state(job_id, status="failed", result=None, error=str(exc.detail))
        except Exception as exc:
            msg = str(exc)
            if "cancelled" in msg.lower():
                _append_job_event(job_id, "cancelled", msg, 100)
                _set_job_state(job_id, status="cancelled", result=None, error=msg)
            else:
                _append_job_event(job_id, "error", msg, 100)
                _set_job_state(job_id, status="failed", result=None, error=msg)

    Thread(target=worker, daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/forecast/jobs/{job_id}")
def forecast_job_status(job_id: str, after: int = -1):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            row = _get_job_row(job_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Job not found")
            try:
                req_json = json.loads(row["request_json"] or "{}")
            except Exception:
                req_json = {}
            try:
                events_json = json.loads(row["events_json"] or "[]")
            except Exception:
                events_json = []
            try:
                result_json = json.loads(row["result_json"]) if row["result_json"] else None
            except Exception:
                result_json = None
            job = {
                "id": str(row["id"]),
                "status": str(row["status"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "request": req_json,
                "events": events_json,
                "result": result_json,
                "error": row["error_text"],
                "cancel_requested": bool(int(row["cancel_requested"] or 0)),
            }
            JOBS[job_id] = job
        events = [e for e in job.get("events", []) if int(e.get("idx", -1)) > after]
        payload = {
            "id": job["id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "events": events,
            "error": job.get("error"),
            "result": job.get("result") if job["status"] == "completed" else None,
        }
    return payload


@app.delete("/api/forecast/jobs/{job_id}")
def forecast_job_cancel(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            row = _get_job_row(job_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Job not found")
            raise HTTPException(status_code=409, detail="Job not loaded; retry status endpoint first")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"ok": True, "status": job["status"]}
        job["cancel_requested"] = True
        _persist_job(job)
    _append_job_event(job_id, "cancel", "Cancellation requested", 100)
    return {"ok": True, "status": "cancelling"}


@app.post("/api/runs/save")
def save_run_endpoint(body: SaveRunRequest):
    save_run(req=body.request, response=body.response, source_note=body.source_note or "Manual save")
    return {"ok": True, "runs": fetch_recent_runs(limit=20)}


@app.get("/api/runs")
def list_saved_runs():
    return {"runs": fetch_recent_runs(limit=50)}


@app.delete("/api/runs/{run_id}")
def delete_saved_run(run_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    return {"ok": True, "runs": fetch_recent_runs(limit=50)}


@app.get("/api/runs/{run_id}")
def replay_run(run_id: int):
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, pair, timeframe, history, model, exchange_mode, horizon_steps, response_json
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        saved = json.loads(row["response_json"] or "{}")
    except Exception:
        saved = {}
    if not isinstance(saved, dict) or not saved.get("forecast") or not saved.get("forecast", {}).get("projection"):
        raise HTTPException(status_code=422, detail="This run was saved before snapshot support and cannot be replayed")

    pair = str(row["pair"])
    timeframe = str(row["timeframe"])
    history = str(row["history"])
    exchange_mode = str(row["exchange_mode"])

    step = infer_step_delta(timeframe)
    days_by_history = {"1d": 1, "2d": 2, "3d": 3, "7d": 7, "15d": 15, "30d": 30, "60d": 60, "90d": 90, "custom": 90}
    base_days = days_by_history.get(history, 30)
    created_at_raw = str(row["created_at"] or "")
    try:
        created_at_dt = datetime.fromisoformat(created_at_raw)
        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
        age_days = max(0, int((datetime.now(timezone.utc) - created_at_dt).total_seconds() / 86400))
    except Exception:
        age_days = 0
    required_days = max(base_days, age_days + 2)
    # Replay should include the original context window before the run timestamp.
    # Use the max between configured history and elapsed time so the blue line is meaningful.
    if "created_at_dt" in locals():
        context_lookback_days = max(base_days, required_days)
        start_time = created_at_dt - timedelta(days=context_lookback_days)
    else:
        start_time = None
    end_time = datetime.now(timezone.utc)

    data, source_note = load_history_for_run_context(
        pair,
        timeframe,
        history,
        exchange_mode,
        min_days=required_days,
        start_time=start_time,
        end_time=end_time,
    )

    target_points = int((required_days * 24 * 3600) / max(1.0, step.total_seconds()))
    target_points = max(80, min(2000, target_points))
    candles = []
    for ts, r in data.tail(target_points).iterrows():
        candles.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": float(r.get("Volume", 0.0) or 0.0),
            }
        )

    payload = saved if isinstance(saved, dict) else {}
    payload.setdefault("forecast", {})
    payload["forecast"]["candles"] = candles

    payload.setdefault("logs", [])
    payload["logs"] = [
        {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%SZ"),
            "level": "info",
            "title": "Run replay loaded",
            "detail": f"Run #{run_id} from {row['created_at']} · updated history from {source_note}",
        }
    ] + payload["logs"][:3]
    payload["runs"] = fetch_recent_runs(limit=20)
    payload["replay"] = {
        "runId": run_id,
        "createdAt": row["created_at"],
        "pair": pair,
        "timeframe": timeframe,
        "history": history,
        "model": row["model"],
        "exchange": exchange_mode,
        "horizon_steps": int(row["horizon_steps"]),
    }
    payload["request_context"] = payload.get("request_context") or {
        "pair": pair,
        "timeframe": timeframe,
        "history": history,
        "model": row["model"],
        "exchange": exchange_mode,
        "horizon_steps": int(row["horizon_steps"]),
    }
    return JSONResponse(content=payload)


# Serve the frontend from repo root
app.mount("/", StaticFiles(directory="/app", html=True), name="static")
