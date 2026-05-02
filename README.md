# Kronos WebUI

UI técnica de Kronos conectada a backend real en Docker.

## Qué incluye

- Frontend terminal (`index.html`) con:
  - input de par (`BTC/USDT`, etc.)
  - selector de timeframe
  - selector de modelo (`Kronos-small`, `Kronos-base`, `Kronos-ensemble`)
  - botón `Run Forecast`
- API FastAPI (`/api/forecast`) que:
  - obtiene histórico desde Yahoo Finance (yfinance)
  - ejecuta inferencia con Kronos oficial
  - devuelve payload para renderizar gráficos, señales, runs y logs
- Despliegue completo con Docker Compose.

## Modelos

- `Kronos-small` -> `NeoQuasar/Kronos-small`
- `Kronos-base` -> `NeoQuasar/Kronos-base`
- `Kronos-ensemble` -> media de `small` + `base`

Nota: `Kronos-large` no es público en Hugging Face en este momento.

## Arranque

```bash
docker compose up -d --build
```

## URL

- WebUI/API: `http://<IP_SERVIDOR>:18080`
- Health: `http://<IP_SERVIDOR>:18080/api/health`

## Endpoint principal

`POST /api/forecast`

Ejemplo:

```bash
curl -X POST http://127.0.0.1:18080/api/forecast \
  -H 'content-type: application/json' \
  -d '{
    "pair": "BTC/USDT",
    "timeframe": "1h",
    "history": "7d",
    "model": "Kronos-base",
    "exchange": "Binance",
    "horizon_steps": 12
  }'
```

## Notas operativas

- Si Yahoo Finance limita peticiones temporalmente, el backend usa fallback para no romper la UI.
- La primera inferencia puede tardar más por descarga/cache de pesos.
