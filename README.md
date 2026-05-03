# Kronos WebUI

Production-ready web terminal for Kronos forecasting with a real FastAPI backend, Dockerized deployment, probabilistic inference, replayable runs, and online backtesting.

## Metadata

- **Description:** Web UI + API for Kronos probabilistic forecasting with saved-run replay and backtesting.
- **Suggested Website:** `http://<SERVER_IP>:18080`
- **Suggested Topics:** `kronos`, `forecasting`, `timeseries`, `crypto`, `fastapi`, `docker`, `binance`, `yfinance`, `sqlite`, `echarts`
- **Primary Language:** `Python + JavaScript`

## Dashboard

![Kronos WebUI Dashboard](assets/dashboard-btc.png)

## Features

- Real market history from `Binance` (default) + `Yahoo Finance`.
- Kronos model execution:
  - `Kronos-mini`
  - `Kronos-small`
  - `Kronos-base`
  - `Kronos-ensemble`
- Probabilistic forecast bands (`p10 / p50 / p90`) from stochastic sampling.
- Adaptive context sizing based on market volatility regime.
- Online backtesting for saved runs once forecast horizon expires.
- Adjustable runtime controls:
  - pair (`BTCUSDT`, etc.)
  - timeframe
  - range
  - horizon
  - sample runs
  - model
  - exchange mode
- Live progress terminal while inference runs.
- Manual `Save` flow for clean `Saved runs` (no auto-save spam).
- Replay any saved run on the main chart with updated historical context.

## Stack

- Frontend: static `index.html` (dark terminal-style UI)
- Backend: `FastAPI`
- Forecast engine: official Kronos models
- Storage: SQLite (`./data/runs.sqlite3`, persistent volume)
- Deployment: Docker Compose

## Run

```bash
docker compose up -d --build
```

## Access

- App: `http://<SERVER_IP>:18080`
- Health: `http://<SERVER_IP>:18080/api/health`

## Core API

- `POST /api/forecast` - forecast response
- `POST /api/forecast/stream` - live progress stream + result
- `POST /api/runs/save` - manual save current forecast
- `GET /api/runs` - list saved runs
- `GET /api/runs/{id}` - replay saved run

Example:

```bash
curl -X POST http://127.0.0.1:18080/api/forecast \
  -H 'content-type: application/json' \
  -d '{
    "pair": "BTCUSDT",
    "timeframe": "1h",
    "history": "15d",
    "model": "Kronos-mini",
    "exchange": "binance",
    "horizon_steps": 24,
    "sample_runs": 30
  }'
```

## Notes

- First run can be slower due to model warm-up.
- Higher `sample_runs` means better probabilistic stability but much higher latency.
- Data persistence is volume-backed (`./data`) so saved runs survive container rebuilds.
