from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import settings
from utils.logging_utils import configure_logging

_db_processed_cache: int | None = None
_db_processed_cache_t: float = 0.0


def _get_processed_count_from_db() -> int:
    """Source de vérité pour l’UI : aligné sur la table, même après purge SQL."""
    global _db_processed_cache, _db_processed_cache_t
    now = time.time()
    ttl = max(0.5, float(getattr(settings, "monitor_broadcast_seconds", 1.0) or 1.0) * 2)
    if _db_processed_cache is not None and now - _db_processed_cache_t < ttl:
        return _db_processed_cache
    try:
        from database import DatabaseClient

        client = DatabaseClient(use_pool=False)
        try:
            n = client.count_processed_total()
        finally:
            client.close()
        _db_processed_cache = n
        _db_processed_cache_t = now
        return n
    except Exception:
        return _db_processed_cache if _db_processed_cache is not None else 0


def _snapshot(
    stats: dict[str, Any],
    worker_count: int,
    pipeline_start: float,
    prev: dict[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    with stats["lock"]:
        processed = _get_processed_count_from_db()
        failed = stats["failed"].value
        skipped = stats["skipped"].value
        # Shared counters: macOS multiprocessing.Queue.qsize() is often unusable
        ingest_sz = stats["ingest_pending"].value
        result_sz = stats["result_pending"].value
        activity = list(stats.get("activity") or [])[-120:]
    files_per_min = None
    if prev is not None and now > prev["t"]:
        dp = processed - prev["processed"]
        dt = now - prev["t"]
        files_per_min = (dp / dt) * 60.0
    return {
        "t": now,
        "uptime_sec": round(now - pipeline_start, 1),
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "ingest_queue": ingest_sz,
        "result_queue": result_sz,
        "queue_max": settings.queue_max_size,
        "workers": worker_count,
        "files_per_min": None if files_per_min is None else round(files_per_min, 2),
        "activity": activity,
    }


def create_app(
    stop_event,
    stats: dict[str, Any],
    worker_count: int,
    pipeline_start: float,
) -> FastAPI:
    app = FastAPI(title="B/L Pipeline Monitor", version="1.0.0")
    history: deque[dict[str, Any]] = deque(maxlen=120)
    prev_snap: dict[str, Any] | None = None

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        # Template uses doubled braces `{{`/`}}` (legacy str.format escape). We only
        # substitute host/port with .replace — collapse braces or JS throws SyntaxError.
        html = (
            _DASHBOARD_HTML.replace("__HOST__", settings.monitor_host)
            .replace("__PORT__", str(settings.monitor_port))
            .replace("{{", "{")
            .replace("}}", "}")
        )
        return HTMLResponse(
            content=html,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/snapshot")
    async def api_snapshot() -> dict[str, Any]:
        nonlocal prev_snap
        now = time.time()
        snap = _snapshot(stats, worker_count, pipeline_start, prev_snap, now)
        rate = snap["files_per_min"]
        prev_snap = {"t": now, "processed": snap["processed"]}
        return {
            "uptime_sec": snap["uptime_sec"],
            "processed": snap["processed"],
            "failed": snap["failed"],
            "skipped": snap["skipped"],
            "ingest_queue": snap["ingest_queue"],
            "result_queue": snap["result_queue"],
            "queue_max": snap["queue_max"],
            "workers": snap["workers"],
            "files_per_min": rate,
            "activity": snap.get("activity") or [],
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        local_prev: dict[str, Any] | None = None
        try:
            while not stop_event.is_set():
                now = time.time()
                snap = _snapshot(stats, worker_count, pipeline_start, local_prev, now)
                rate = snap.get("files_per_min")
                local_prev = {"t": now, "processed": snap["processed"]}
                payload = {
                    "uptime_sec": snap["uptime_sec"],
                    "processed": snap["processed"],
                    "failed": snap["failed"],
                    "skipped": snap["skipped"],
                    "ingest_queue": snap["ingest_queue"],
                    "result_queue": snap["result_queue"],
                    "queue_max": snap["queue_max"],
                    "workers": snap["workers"],
                    "files_per_min": rate,
                    "activity": snap.get("activity") or [],
                }
                history.append(
                    {"t": now, "processed": snap["processed"], "rate": rate or 0}
                )
                payload["history"] = list(history)
                await websocket.send_json(payload)
                await asyncio.sleep(settings.monitor_broadcast_seconds)
        except WebSocketDisconnect:
            return

    return app


def run_monitor_server(
    stop_event,
    stats: dict[str, Any],
    worker_count: int,
) -> None:
    """Run Uvicorn in the current thread (use from a background thread in MainProcess).

    A separate *multiprocessing* process is unreliable here: Manager proxies + uvloop on
    macOS often prevents the HTTP server from accepting connections. Same-process thread works.
    """
    import logging
    import traceback

    configure_logging()
    log = logging.getLogger("bl_pipeline")
    pipeline_start = time.time()
    app = create_app(stop_event, stats, worker_count, pipeline_start)
    config = uvicorn.Config(
        app,
        host=settings.monitor_host,
        port=settings.monitor_port,
        log_level="info",
        loop="asyncio",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def shutdown_when_stopped() -> None:
        stop_event.wait()
        server.should_exit = True

    threading.Thread(target=shutdown_when_stopped, daemon=True).start()
    try:
        log.info(
            "Monitor binding http://%s:%s",
            settings.monitor_host,
            settings.monitor_port,
        )
        server.run()
    except OSError as exc:
        log.error("Monitor could not bind to port %s: %s", settings.monitor_port, exc)
        raise
    except Exception:
        log.error("Monitor crashed:\n%s", traceback.format_exc())
        raise


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>B/L Pipeline — Monitoring</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-deep: #0c0b1a;
      --text: #f4f6ff;
      --muted: #9aa3c2;
      --glass: rgba(255,255,255,0.06);
      --glass-border: rgba(255,255,255,0.12);
      --green: #34d399;
      --green-dim: #059669;
      --rose: #fb7185;
      --rose-dim: #e11d48;
      --amber: #fbbf24;
      --amber-dim: #d97706;
      --cyan: #22d3ee;
      --cyan-dim: #0891b2;
      --violet: #a78bfa;
      --violet-dim: #7c3aed;
      --indigo: #818cf8;
    }}
    * {{ box-sizing: border-box; }}
    html {{ -webkit-font-smoothing: antialiased; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Poppins", system-ui, -apple-system, sans-serif;
      color: var(--text);
      background: var(--bg-deep);
      background-image:
        radial-gradient(ellipse 120% 80% at 10% -20%, rgba(167,139,250,0.35), transparent 50%),
        radial-gradient(ellipse 90% 60% at 100% 0%, rgba(34,211,238,0.2), transparent 45%),
        radial-gradient(ellipse 70% 50% at 50% 100%, rgba(52,211,153,0.15), transparent 50%);
      background-attachment: fixed;
    }}
    .wrap {{ max-width: 1220px; margin: 0 auto; padding: 1.5rem 1.25rem 2.5rem; }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.75rem;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 1rem;
    }}
    .logo {{
      width: 52px; height: 52px;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--violet), var(--cyan));
      display: grid; place-items: center;
      font-size: 1.5rem;
      box-shadow: 0 8px 32px rgba(124,58,237,0.35);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.15rem, 2.5vw, 1.45rem);
      font-weight: 700;
      letter-spacing: -0.02em;
      line-height: 1.25;
    }}
    .subtitle {{ margin: 0.25rem 0 0; font-size: 0.85rem; color: var(--muted); font-weight: 500; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: var(--glass);
      border: 1px solid var(--glass-border);
      color: var(--muted);
    }}
    .dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--amber); animation: pulse 2s ease-in-out infinite; box-shadow: 0 0 12px var(--amber); }}
    @keyframes pulse {{ 0%,100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.55; transform: scale(0.92); }} }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
      gap: 1rem;
    }}
    .metric {{
      position: relative;
      padding: 1.15rem 1.2rem 1.2rem;
      border-radius: 18px;
      background: var(--glass);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--glass-border);
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }}
    .metric::before {{
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      border-radius: 18px 18px 0 0;
    }}
    .metric.processed::before {{ background: linear-gradient(90deg, var(--green), var(--cyan)); }}
    .metric.failed::before {{ background: linear-gradient(90deg, var(--rose), var(--amber)); }}
    .metric.skipped::before {{ background: linear-gradient(90deg, var(--amber), var(--rose-dim)); }}
    .metric.rate::before {{ background: linear-gradient(90deg, var(--cyan), var(--indigo)); }}
    .metric.neutral::before {{ background: linear-gradient(90deg, var(--violet), var(--violet-dim)); }}
    .metric-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }}
    .metric h3 {{
      margin: 0;
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .metric .ico {{
      width: 36px;
      height: 36px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      font-size: 1.1rem;
    }}
    .metric.processed .ico {{ background: rgba(52,211,153,0.2); }}
    .metric.failed .ico {{ background: rgba(251,113,133,0.2); }}
    .metric.skipped .ico {{ background: rgba(251,191,36,0.2); }}
    .metric.rate .ico {{ background: rgba(34,211,238,0.2); }}
    .metric.neutral .ico {{ background: rgba(129,140,248,0.2); }}
    .metric .val {{
      font-size: 1.85rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }}
    .metric.processed .val {{ color: var(--green); text-shadow: 0 0 24px rgba(52,211,153,0.35); }}
    .metric.failed .val {{ color: var(--rose); text-shadow: 0 0 24px rgba(251,113,133,0.35); }}
    .metric.skipped .val {{ color: var(--amber); text-shadow: 0 0 24px rgba(251,191,36,0.25); }}
    .metric.rate .val {{ color: var(--cyan); text-shadow: 0 0 24px rgba(34,211,238,0.35); }}
    .metric.neutral .val {{ color: #c7d2fe; }}
    .panel {{
      margin-top: 1.35rem;
      padding: 1.35rem 1.4rem;
      border-radius: 20px;
      background: var(--glass);
      backdrop-filter: blur(12px);
      border: 1px solid var(--glass-border);
      box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }}
    .panel h2 {{
      margin: 0 0 1rem;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .q-row {{ margin-bottom: 1.15rem; }}
    .q-row:last-child {{ margin-bottom: 0; }}
    .q-label {{ font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.45rem; }}
    .bar-wrap {{
      height: 14px;
      border-radius: 999px;
      background: rgba(0,0,0,0.35);
      overflow: hidden;
      border: 1px solid var(--glass-border);
    }}
    .bar {{
      height: 100%;
      border-radius: 999px;
      transition: width 0.35s ease;
      box-shadow: 0 0 20px rgba(34,211,238,0.35);
    }}
    .bar.ingest {{ background: linear-gradient(90deg, var(--cyan-dim), var(--cyan)); }}
    .bar.result {{ background: linear-gradient(90deg, var(--violet-dim), var(--violet)); box-shadow: 0 0 20px rgba(167,139,250,0.4); }}
    .q-meta {{ margin: 0.4rem 0 0; font-size: 0.82rem; color: var(--muted); font-weight: 500; }}
    .chart-box {{ margin-top: 1.35rem; }}
    .chart-inner {{
      border-radius: 20px;
      padding: 1.1rem 1rem 0.75rem;
      background: var(--glass);
      backdrop-filter: blur(12px);
      border: 1px solid var(--glass-border);
      height: 300px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }}
    .chart-title {{ font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 0 0 0.5rem 0.25rem; }}
    footer {{
      text-align: center;
      margin-top: 2rem;
      font-size: 0.78rem;
      color: var(--muted);
      font-weight: 500;
    }}
    footer code {{
      padding: 0.2rem 0.45rem;
      border-radius: 8px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--glass-border);
      color: var(--cyan);
      font-family: ui-monospace, monospace;
      font-size: 0.85em;
    }}
    .error-msg {{ color: var(--rose); padding: 1rem; text-align: center; font-weight: 600; }}
    .journal-above-queue {{ margin-top: 1.35rem; width: 100%; }}
    .live-wrap {{ margin-top: 1.35rem; display: grid; gap: 1.25rem; }}
    @media (min-width: 960px) {{ .live-wrap {{ grid-template-columns: 1fr 1fr; align-items: start; }} }}
    .live-panel {{
      padding: 1.25rem 1.35rem;
      border-radius: 20px;
      background: var(--glass);
      backdrop-filter: blur(14px);
      border: 1px solid var(--glass-border);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
      min-height: 120px;
    }}
    .live-panel h2 {{
      margin: 0 0 1rem;
      font-size: 0.88rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .live-panel h2 .badge {{
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      background: rgba(34,211,238,0.15);
      color: var(--cyan);
      border: 1px solid rgba(34,211,238,0.25);
    }}
    .proc-grid {{ display: flex; flex-wrap: wrap; gap: 0.85rem; }}
    .proc-empty {{
      font-size: 0.85rem;
      color: var(--muted);
      font-weight: 500;
      padding: 0.5rem 0;
    }}
    .proc-card {{
      flex: 1 1 240px;
      max-width: 100%;
      padding: 1rem 1.15rem;
      border-radius: 16px;
      border: 1px solid rgba(167,139,250,0.35);
      background: linear-gradient(
        135deg,
        rgba(124,58,237,0.22) 0%,
        rgba(34,211,238,0.12) 50%,
        rgba(124,58,237,0.18) 100%
      );
      background-size: 220% 220%;
      animation: cardGlow 4s ease-in-out infinite;
      position: relative;
      overflow: hidden;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset, 0 12px 40px rgba(124,58,237,0.15);
    }}
    .proc-card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(
        110deg,
        transparent 0%,
        rgba(255,255,255,0.07) 45%,
        rgba(255,255,255,0.14) 50%,
        rgba(255,255,255,0.07) 55%,
        transparent 100%
      );
      background-size: 220% 100%;
      animation: shimmer 2.4s ease-in-out infinite;
      pointer-events: none;
    }}
    @keyframes shimmer {{
      0% {{ background-position: 200% 0; }}
      100% {{ background-position: -200% 0; }}
    }}
    @keyframes cardGlow {{
      0%, 100% {{ background-position: 0% 50%; }}
      50% {{ background-position: 100% 50%; }}
    }}
    .proc-card .fn {{
      position: relative;
      z-index: 1;
      font-weight: 600;
      font-size: 0.9rem;
      color: var(--text);
      margin-bottom: 0.65rem;
      word-break: break-all;
    }}
    .proc-card .meta {{
      position: relative;
      z-index: 1;
      font-size: 0.72rem;
      color: var(--muted);
      font-weight: 500;
    }}
    .dots {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
      vertical-align: middle;
      margin-left: 0.35rem;
    }}
    .dots span {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 10px var(--cyan);
      animation: dotBounce 1.15s ease-in-out infinite;
    }}
    .dots span:nth-child(2) {{ animation-delay: 0.15s; }}
    .dots span:nth-child(3) {{ animation-delay: 0.3s; }}
    @keyframes dotBounce {{
      0%, 80%, 100% {{ transform: translateY(0) scale(0.75); opacity: 0.35; }}
      40% {{ transform: translateY(-5px) scale(1); opacity: 1; }}
    }}
    .wait-card {{
      flex: 1 1 200px;
      padding: 0.85rem 1rem;
      border-radius: 14px;
      border: 1px solid var(--glass-border);
      background: rgba(255,255,255,0.04);
    }}
    .wait-card .pulse-ring {{
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--amber);
      margin-right: 0.45rem;
      box-shadow: 0 0 0 0 rgba(251,191,36,0.55);
      animation: ringPulse 1.8s ease-out infinite;
    }}
    @keyframes ringPulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(251,191,36,0.5); }}
      70% {{ box-shadow: 0 0 0 10px rgba(251,191,36,0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(251,191,36,0); }}
    }}
    .log-scroll {{
      max-height: 340px;
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 0.25rem;
      margin: 0 -0.15rem;
      scrollbar-width: thin;
      scrollbar-color: rgba(167,139,250,0.45) transparent;
    }}
    .log-scroll::-webkit-scrollbar {{ width: 6px; }}
    .log-scroll::-webkit-scrollbar-thumb {{
      background: rgba(167,139,250,0.4);
      border-radius: 99px;
    }}
    .log-line {{
      display: grid;
      grid-template-columns: 4.2rem 1fr;
      gap: 0.65rem 0.85rem;
      align-items: start;
      padding: 0.55rem 0.65rem;
      margin-bottom: 0.35rem;
      border-radius: 12px;
      background: rgba(0,0,0,0.2);
      border: 1px solid rgba(255,255,255,0.05);
      font-size: 0.8rem;
      animation: logIn 0.35s ease-out;
    }}
    @keyframes logIn {{
      from {{ opacity: 0; transform: translateX(-6px); }}
      to {{ opacity: 1; transform: translateX(0); }}
    }}
    .log-line .ts {{ font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.72rem; font-weight: 600; padding-top: 0.15rem; }}
    .log-line .body {{ min-width: 0; }}
    .log-line .lbl {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-weight: 600;
      margin-bottom: 0.2rem;
    }}
    .log-line .file {{ color: var(--text); word-break: break-all; font-weight: 500; }}
    .log-line .detail {{ color: var(--muted); font-size: 0.74rem; margin-top: 0.15rem; line-height: 1.35; }}
    .tag {{ font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; padding: 0.15rem 0.45rem; border-radius: 6px; }}
    .tag.q {{ background: rgba(251,191,36,0.18); color: var(--amber); border: 1px solid rgba(251,191,36,0.3); }}
    .tag.p {{ background: rgba(34,211,238,0.18); color: var(--cyan); border: 1px solid rgba(34,211,238,0.3); }}
    .tag.d {{ background: rgba(52,211,153,0.18); color: var(--green); border: 1px solid rgba(52,211,153,0.25); }}
    .tag.f {{ background: rgba(251,113,133,0.18); color: var(--rose); border: 1px solid rgba(251,113,133,0.3); }}
    .tag.s {{ background: rgba(129,140,248,0.2); color: #c7d2fe; border: 1px solid rgba(129,140,248,0.3); }}
    .tag.r {{ background: rgba(244,114,182,0.15); color: #f9a8d4; border: 1px solid rgba(244,114,182,0.25); }}
    .tag.sk {{ background: rgba(251,191,36,0.12); color: #fcd34d; border: 1px solid rgba(251,191,36,0.2); }}
  </style>
</head>
<body style="margin:0;font-family:'Poppins',system-ui,sans-serif;background:#0c0b1a;color:#f4f6ff;">
  <div class="wrap">
    <header>
      <div class="brand">
        <div class="logo" aria-hidden="true">📄</div>
        <div>
          <h1>Connaissement (B/L)</h1>
          <p class="subtitle">Surveillance temps réel · Pipeline documents</p>
        </div>
      </div>
      <div class="pill"><span class="dot" id="liveDot"></span><span id="connLabel">Connexion…</span></div>
    </header>
    <div id="err" class="error-msg" style="display:none;"></div>
    <div class="grid">
      <div class="metric processed">
        <div class="metric-top"><h3>Traités</h3><span class="ico">✓</span></div>
        <div class="val" id="processed">0</div>
      </div>
      <div class="metric failed">
        <div class="metric-top"><h3>Échecs</h3><span class="ico">⚠</span></div>
        <div class="val" id="failed">0</div>
      </div>
      <div class="metric skipped">
        <div class="metric-top"><h3>Doublons / ignorés</h3><span class="ico">◇</span></div>
        <div class="val" id="skipped">0</div>
      </div>
      <div class="metric rate">
        <div class="metric-top"><h3>Débit (approx. / min)</h3><span class="ico">⚡</span></div>
        <div class="val" id="rate">—</div>
      </div>
      <div class="metric neutral">
        <div class="metric-top"><h3>Uptime</h3><span class="ico">⏱</span></div>
        <div class="val" id="uptime">0s</div>
      </div>
      <div class="metric neutral">
        <div class="metric-top"><h3>Workers</h3><span class="ico">⊛</span></div>
        <div class="val" id="workers">—</div>
      </div>
    </div>
    <div class="live-panel journal-above-queue">
      <h2>Journal des actions <span class="badge" id="logBadge" style="background:rgba(167,139,250,0.15);color:var(--violet);border-color:rgba(167,139,250,0.25);">temps réel</span></h2>
      <div id="activityLog" class="log-scroll"></div>
    </div>
    <div class="live-wrap">
      <div class="live-panel">
        <h2>Traitement en cours <span class="badge" id="procBadge">0</span></h2>
        <div id="processingCards" class="proc-grid"></div>
      </div>
      <div class="live-panel">
        <h2>En attente <span class="badge" id="waitBadge" style="background:rgba(251,191,36,0.15);color:var(--amber);border-color:rgba(251,191,36,0.25);">0</span></h2>
        <div id="waitingCards" class="proc-grid"></div>
      </div>
    </div>
    <div class="panel">
      <h2>Files d’attente</h2>
      <div class="q-row">
        <div class="q-label">Entrée (ingest)</div>
        <div class="bar-wrap"><div class="bar ingest" id="barIngest" style="width:0%"></div></div>
        <p class="q-meta"><span id="ingestN">0</span> / <span id="qmax">0</span></p>
      </div>
      <div class="q-row">
        <div class="q-label">Vers MySQL (résultats)</div>
        <div class="bar-wrap"><div class="bar result" id="barResult" style="width:0%"></div></div>
        <p class="q-meta"><span id="resultN">0</span> / <span id="qmax2">0</span></p>
      </div>
    </div>
    <div class="chart-box">
      <div class="chart-inner">
        <div class="chart-title">Débit instantané</div>
        <canvas id="chartRate"></canvas>
      </div>
    </div>
    <footer>API <code>/api/snapshot</code> · WebSocket <code>/ws</code> · <code>http://__HOST__:__PORT__</code></footer>
  </div>
  <script>
    const fmtUptime = (s) => {{
      if (s < 60) return Math.round(s) + 's';
      const m = Math.floor(s / 60), sec = Math.round(s % 60);
      return m + 'm ' + sec + 's';
    }};
    let chart;
    const labels = [];
    const rates = [];
    function initChart() {{
      const ctx = document.getElementById('chartRate').getContext('2d');
      chart = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: labels,
          datasets: [{{
            label: 'Fichiers / min (instantané)',
            data: rates,
            borderColor: '#22d3ee',
            backgroundColor: 'rgba(34,211,238,0.2)',
            fill: true,
            tension: 0.35,
            pointRadius: 0,
            borderWidth: 2,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ display: true, grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ maxTicksLimit: 8, color: '#9aa3c2' }} }},
            y: {{ beginAtZero: true, grid: {{ color: 'rgba(255,255,255,0.06)' }}, ticks: {{ color: '#9aa3c2' }} }}
          }}
        }}
      }});
    }}
    function setBar(el, n, max) {{
      const p = max > 0 ? Math.min(100, (n / max) * 100) : 0;
      el.style.width = p + '%';
    }}
    function mergeJobStates(activity) {{
      const sorted = [...activity].sort((a, b) => a.t - b.t);
      const map = new Map();
      for (const ev of sorted) {{
        const jid = ev.job_id || '';
        const key = jid || ('anon:' + ev.file + ':' + ev.t);
        const prev = map.get(key) || {{}};
        map.set(key, {{
          job_id: ev.job_id || prev.job_id,
          file: ev.file || prev.file,
          lastKind: ev.kind,
          lastT: ev.t,
          worker_id: ev.worker_id != null ? ev.worker_id : prev.worker_id,
        }});
      }}
      return Array.from(map.values()).filter((j) => j.file);
    }}
    function tagForKind(kind) {{
      const m = {{
        queued: ['q', 'File'],
        processing: ['p', 'Traitement'],
        done: ['d', 'Terminé'],
        saved: ['s', 'Base'],
        failed: ['f', 'Échec'],
        retry: ['r', 'Retry'],
        skipped_duplicate: ['sk', 'Doublon'],
        already_done: ['s', 'Déjà traité'],
        missing_after_retries: ['f', 'Introuvable'],
        failed_no_db: ['f', 'Échec'],
      }};
      const x = m[kind] || ['s', String(kind)];
      return {{ cls: x[0], label: x[1] }};
    }}
    function fmtClock(t) {{
      const d = new Date(t * 1000);
      return d.toLocaleTimeString('fr-FR', {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
    }}
    function renderLiveActivity(activity) {{
      const act = activity || [];
      const jobs = mergeJobStates(act);
      const processing = jobs.filter((j) => j.lastKind === 'processing');
      const waiting = jobs.filter((j) => j.lastKind === 'queued' || j.lastKind === 'retry');
      const procEl = document.getElementById('processingCards');
      const waitEl = document.getElementById('waitingCards');
      document.getElementById('procBadge').textContent = String(processing.length);
      document.getElementById('waitBadge').textContent = String(waiting.length);
      if (processing.length === 0) {{
        procEl.innerHTML = '<p class="proc-empty">Aucun fichier en cours d’extraction.</p>';
      }} else {{
        procEl.innerHTML = processing.map((j) => {{
          const w = j.worker_id != null ? 'Worker ' + j.worker_id : '';
          return '<div class="proc-card" role="status">'
            + '<div class="fn">' + (j.file || '') + '</div>'
            + '<div class="meta">' + w + ' · IA + OCR<span class="dots" aria-hidden="true"><span></span><span></span><span></span></span></div>'
            + '</div>';
        }}).join('');
      }}
      if (waiting.length === 0) {{
        waitEl.innerHTML = '<p class="proc-empty">File d’attente vide.</p>';
      }} else {{
        waitEl.innerHTML = waiting.map((j) => {{
          const lbl = j.lastKind === 'retry' ? 'Nouvel essai' : 'En file';
          return '<div class="wait-card"><span class="pulse-ring"></span><strong>' + (j.file || '') + '</strong><div class="meta" style="margin-top:0.35rem">' + lbl + '</div></div>';
        }}).join('');
      }}
      const logEl = document.getElementById('activityLog');
      const lines = [...act].sort((a, b) => b.t - a.t).slice(0, 80);
      logEl.innerHTML = lines.map((ev) => {{
        const tg = tagForKind(ev.kind);
        let detail = '';
        if (ev.seconds != null) detail += ev.seconds + ' s';
        if (ev.worker_id != null) detail += (detail ? ' · ' : '') + 'Worker ' + ev.worker_id;
        if (ev.attempt != null) detail += (detail ? ' · ' : '') + 'essai ' + ev.attempt;
        if (ev.error) detail += (detail ? ' — ' : '') + ev.error;
        return '<div class="log-line"><div class="ts">' + fmtClock(ev.t) + '</div><div class="body">'
          + '<div class="lbl"><span class="tag ' + tg.cls + '">' + tg.label + '</span></div>'
          + '<div class="file">' + (ev.file || '') + '</div>'
          + (detail ? '<div class="detail">' + detail + '</div>' : '')
          + '</div></div>';
      }}).join('');
    }}
    function connect() {{
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(proto + '//' + location.host + '/ws');
      const dot = document.getElementById('liveDot');
      const conn = document.getElementById('connLabel');
      const errEl = document.getElementById('err');
      ws.onopen = () => {{ errEl.style.display = 'none'; conn.textContent = 'Temps réel'; conn.style.color = '#34d399'; dot.style.background = '#34d399'; dot.style.boxShadow = '0 0 14px #34d399'; }};
      ws.onclose = () => {{ conn.textContent = 'Déconnecté — reconnexion…'; conn.style.color = '#fbbf24'; dot.style.background = '#fbbf24'; dot.style.boxShadow = '0 0 14px #fbbf24'; setTimeout(connect, 2000); }};
      ws.onerror = () => {{ errEl.textContent = 'Erreur WebSocket'; errEl.style.display = 'block'; }};
      ws.onmessage = (ev) => {{
        const d = JSON.parse(ev.data);
        document.getElementById('processed').textContent = d.processed;
        document.getElementById('failed').textContent = d.failed;
        document.getElementById('skipped').textContent = d.skipped;
        document.getElementById('rate').textContent = d.files_per_min != null ? d.files_per_min : '—';
        document.getElementById('uptime').textContent = fmtUptime(d.uptime_sec);
        document.getElementById('workers').textContent = d.workers;
        const qm = d.queue_max || 1;
        document.getElementById('qmax').textContent = qm;
        document.getElementById('qmax2').textContent = qm;
        const iq = d.ingest_queue != null ? d.ingest_queue : 0;
        const rq = d.result_queue != null ? d.result_queue : 0;
        document.getElementById('ingestN').textContent = iq;
        document.getElementById('resultN').textContent = rq;
        setBar(document.getElementById('barIngest'), iq, qm);
        setBar(document.getElementById('barResult'), rq, qm);
        const h = d.history || [];
        const last = h.slice(-60);
        labels.length = 0; rates.length = 0;
        last.forEach((pt, i) => {{
          labels.push(i);
          rates.push(pt.rate != null ? pt.rate : 0);
        }});
        if (chart) chart.update('none');
        renderLiveActivity(d.activity || []);
      }};
    }}
    initChart();
    connect();
  </script>
</body>
</html>
"""
