"""
FastAPI Application — UAV Aero-Piston Engine Digital Twin.

Architecture
------------
A background asyncio task drives the simulation loop at real-time speed
(one step every ``dt`` seconds).  On each tick the simulator produces
telemetry that is immediately fed to the Digital Twin Core.  The resulting
:class:`TwinState` is kept in a module-level shared-state object and
broadcast to every active WebSocket client.

Exposed surface
---------------
  GET  /                  → React operations dashboard
  GET  /health            → liveness probe (for monitoring)
  GET  /status            → simulation run metadata
  GET  /telemetry         → latest combined twin state (JSON)
  GET  /history?n=<int>   → last *n* twin states (JSON array, max 500)
  POST /reset             → restart the simulation from t = 0
  WS   /ws/telemetry      → push latest twin state on every tick
       /ws/telemetry?interval=<float>  override push interval (seconds)

Runtime
-------
- The simulator emits snapshots to Kinetic Tesla PAMBU Layer 1 through Redis.
- The latest Layer 3/4/7 results are adapted to the existing public API.
- The simulator keeps its original asyncio cadence.
- All shared state is protected with asyncio.Lock to be safe with future
  concurrent endpoint calls.
- CORS is fully open for local dashboard development.
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.simulator import EngineSimulator
from app.kinetic_bridge import KineticTwin as DigitalTwinCore, PipelineState as TwinState
from layer1.adapter import parameter_catalog, validate_overrides
from app.replay import MissionReplayEngine, list_available_missions
from app.maintenance_advisory import draft_with_groq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("twin.api")

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
_SIM_DT: float = 0.5          # existing simulator cadence
_AMBIENT_TEMP: float = 25.0   # °C — ISA standard day
_HISTORY_MAXLEN: int = 500    # maximum states kept in ring buffer

# ---------------------------------------------------------------------------
# Shared runtime state  (module-level singletons, guarded by lock)
# ---------------------------------------------------------------------------
_sim_lock = asyncio.Lock()
_simulator: EngineSimulator | None = None
_twin: DigitalTwinCore | None = None
_latest_state: dict | None = None
_latest_state_received_at: float = 0.0
_control_revision: int = 0

# Mode Control (live vs replay)
_current_mode: str = "live"
_current_mission_name: str | None = None
_replay_engine: MissionReplayEngine | None = None

# Simulation metadata
_sim_meta: dict[str, Any] = {
    "started_at": None,
    "step_count": 0,
    "simulation_time": 0.0,
    "running": False,
    "mode": "live",
    "mission": None,
    "replay_progress_pct": 0.0,
}

# Background task handle
_sim_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class _ConnectionManager:
    """Tracks active WebSocket clients and delivers broadcast messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        log.info("WebSocket connected — %d active client(s)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        log.info("WebSocket disconnected — %d active client(s)", len(self._connections))

    async def broadcast(self, payload: dict) -> None:
        """Send *payload* to all connected clients; drop stale connections."""
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


manager = _ConnectionManager()


# ---------------------------------------------------------------------------
# API response normalisation
# ---------------------------------------------------------------------------

def _to_api_format(state_dict: dict) -> dict:
    """Expose the joined Layer 7 assessment through the original MALE API."""
    return dict(state_dict)


# ---------------------------------------------------------------------------
# Simulation loop  (runs as a background asyncio task)
# ---------------------------------------------------------------------------

async def _simulation_loop() -> None:
    """
    Continuously step the simulator and twin core at real-time speed.

    Each iteration:
      1. Calls ``simulator.step()`` to produce one telemetry packet.
      2. Passes it to ``twin.update()`` to obtain a full :class:`TwinState`.
      3. Serialises the state and stores it as the globally latest value.
      4. Broadcasts the state to all connected WebSocket clients.
      5. Sleeps for ``_SIM_DT`` seconds before the next iteration.
    """
    global _latest_state, _latest_state_received_at

    log.info("Simulation loop starting (dt=%.2f s, ambient=%.1f °C)", _SIM_DT, _AMBIENT_TEMP)
    _sim_meta["running"] = True
    _sim_meta["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        while True:
            async with _sim_lock:
                if _current_mode == "replay" and _replay_engine is not None:
                    telem, is_finished = _replay_engine.get_step()
                    _sim_meta["replay_progress_pct"] = _replay_engine.progress_pct
                    if is_finished:
                        log.info("Mission replay completed for '%s'.", _current_mission_name)
                        _sim_meta["running"] = False
                else:
                    telem = _simulator.step()                      # type: ignore[union-attr]

                telem = dict(telem)
                telem["control_revision"] = _control_revision
                if _simulator is not None:
                    telem["override_values"] = {
                        name: _simulator.override_values[name]
                        for name, enabled in _simulator.override_flags.items() if enabled
                    }
                    provenance = dict(telem.get("provenance") or {})
                    provenance.update(_simulator.control_provenance())
                    if telem.get("atmospheric_profile") is not None:
                        provenance["environment"] = {
                            "source": "simulator_environment_control", "measured": False,
                            "profile": telem["atmospheric_profile"],
                            "altitude_ft": telem.get("altitude"),
                            "ambient_temp_degC": telem.get("ambient_temp"),
                            "control_revision": _control_revision,
                        }
                    if telem.get("fault_active"):
                        provenance["fault_injection"] = {
                            "source": "simulator_fault_injection", "measured": False,
                            "fault": _simulator.injected_fault,
                            "control_revision": _control_revision,
                        }
                    telem["provenance"] = provenance

                state: TwinState = await _twin.update(telem)         # type: ignore[union-attr]
                # Normalise to stable public API format before storing/broadcasting
                payload = _to_api_format(state.to_dict())
                payload["mode"] = _current_mode
                payload["mission"] = _current_mission_name
                _latest_state = payload
                _latest_state_received_at = time.monotonic()
                _sim_meta["step_count"] = _twin.step_count     # type: ignore[union-attr]
                _sim_meta["simulation_time"] = round(state.timestamp, 2)
                _sim_meta["mode"] = _current_mode
                _sim_meta["mission"] = _current_mission_name

            # Broadcast outside the lock so slow clients don't delay simulation
            await manager.broadcast(payload)

            # Log anomaly transitions to the server console
            if state.anomaly_detected:
                log.warning(
                    "ANOMALY t=%.1f s | %s",
                    state.timestamp,
                    state.diagnostic[:80],
                )

            await asyncio.sleep(_SIM_DT)

    except Exception:
        log.exception("Simulation loop failed")
        raise
    except asyncio.CancelledError:
        log.info("Simulation loop cancelled — shutting down cleanly.")
        _sim_meta["running"] = False
        raise
    finally:
        _sim_meta["running"] = False


# ---------------------------------------------------------------------------
# FastAPI lifespan  (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise singletons and start the simulation loop on startup."""
    global _simulator, _twin, _sim_task

    log.info("Starting UAV Engine Digital Twin API...")
    _simulator = EngineSimulator(dt=_SIM_DT, seed=None)   # non-deterministic
    _twin = DigitalTwinCore(ambient_temp=_AMBIENT_TEMP, history_maxlen=_HISTORY_MAXLEN, sample_rate_hz=1 / _SIM_DT)

    _sim_task = asyncio.create_task(_simulation_loop(), name="simulation_loop")

    yield   # ── application is running ──────────────────────────────────────

    log.info("Shutting down UAV Engine Digital Twin API...")
    _sim_task.cancel()
    try:
        await _sim_task
    except asyncio.CancelledError:
        pass
    await _twin.close()
    log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UAV Aero-Piston Engine Digital Twin API",
    description=(
        "Real-time engine telemetry stream, first-principles thermodynamic Digital Twin, "
        "Kinetic Tesla PAMBU Layers 1–7 prediction pipeline, "
        "degradation tracking, conceptual RUL estimation, explainable diagnostics, and mission replay API."
    ),
    version="0.3.0",
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — local API development; the React build uses same-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# React dashboard and API
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"], summary="System info card")
async def root():
    """
    Returns a static description card.
    Useful as a quick liveness check from a browser.
    """
    index = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "index.html"
    if index.is_file():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return {
        "system": "UAV Aero-Piston Engine Digital Twin",
        "version": "0.2.0",
        "description": "Physics-based digital twin with real-time EHI and anomaly detection.",
        "endpoints": {
            "3d_viewer":        "GET /engine/index.html",
            "latest_telemetry": "GET /telemetry",
            "history":          "GET /history?n=<int>",
            "status":           "GET /status",
            "reset":            "POST /reset",
            "websocket":        "WS /ws/telemetry",
            "docs":             "GET /docs",
        },
    }


@app.get("/health", tags=["System"], summary="Liveness probe")
async def health_check() -> dict:
    """Minimal liveness endpoint for load-balancer / container health probes."""
    return {
        "status": "healthy",
        "simulation_running": _sim_meta.get("running", False),
    }


@app.get("/status", tags=["System"], summary="Simulation run metadata")
async def simulation_status() -> dict:
    """
    Returns metadata about the current simulation session:
    step count, wall-clock start time, simulated time elapsed,
    and number of active WebSocket clients.
    """
    return {
        **_sim_meta,
        "websocket_clients": manager.client_count,
        "history_length": len(_twin.history) if _twin else 0,  # type: ignore[arg-type]
    }


@app.get("/telemetry", tags=["Data"], summary="Latest twin state")
async def get_telemetry() -> dict:
    """
    Returns the most recent combined telemetry + Digital Twin state snapshot.

    The response contains:
    - Raw sensor readings (rpm, egt, cht, oil_temp, throttle)
    - Physics-model expected values
    - Per-channel residuals (raw and EMA-smoothed)
    - Engine Health Index (0–100) and status label
    - Anomaly detection results and diagnostic message
    - Ground-truth fault flag (for dashboard validation display)
    """
    if _latest_state is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation has not produced any data yet. Please retry shortly."
        )
    return _latest_state


@app.get("/history", tags=["Data"], summary="Recent twin state history")
async def get_history(
    n: int = Query(default=100, ge=1, le=500, description="Number of recent states to return"),
) -> list[dict]:
    """
    Returns the last *n* Digital Twin state snapshots (most recent last).

    Useful for the dashboard to initialise time-series charts without
    waiting for enough WebSocket frames to accumulate.

    - Maximum 500 entries (the internal ring buffer capacity).
    - States are ordered oldest → newest.
    """
    if _twin is None or len(_twin.history) == 0:
        return []

    async with _sim_lock:
        # Snapshot history under lock to avoid mid-append iteration
        all_states = list(_twin.history)

    # Slice to requested window (most recent n); normalise each state
    selected = all_states[-n:]
    return [_to_api_format(s.to_dict()) for s in selected]


@app.post("/reset", tags=["Control"], summary="Reset simulation to t = 0")
async def reset_simulation() -> dict:
    """
    Restarts both the engine simulator and the digital twin from initial
    conditions (t = 0, all state variables zeroed, EHI = 100).

    WebSocket stream continues without interruption — clients will simply
    see a discontinuity in the timestamp as it jumps back to 0.5 s.
    """
    async with _sim_lock:
        if _simulator is None or _twin is None:
            raise HTTPException(status_code=503, detail="Simulation not yet initialised.")
        _simulator.reset()
        _twin.reset()
        global _control_revision
        _control_revision += 1
        _simulator.control_revision = _control_revision
        _sim_meta["step_count"] = 0
        _sim_meta["simulation_time"] = 0.0
        _sim_meta["started_at"] = datetime.now(timezone.utc).isoformat()

    log.info("Simulation reset requested via POST /reset")
    return {"status": "reset", "message": "Simulation restarted from t = 0."}


@app.post("/simulator/control", tags=["Control"], summary="Set manual simulator controls & fault injection")
async def simulator_control(
    manual_override: bool = Query(default=True, description="Enable or disable manual control mode"),
    throttle: float = Query(default=0.6, ge=0.2, le=0.95, description="Manual throttle position (0.2 - 0.95)"),
    injected_fault: str = Query(default="NONE", description="Fault injection mode (NONE, OVERHEATING, LUBRICATION_ISSUE, EXHAUST_LEAK)"),
) -> dict:
    """
    Manually override simulator throttle and dynamically inject specific live fault symptoms.
    """
    global _current_mode, _current_mission_name, _replay_engine, _control_revision
    async with _sim_lock:
        if _simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not initialised.")
        
        if manual_override:
            # Switch back to live simulation mode if currently in replay mode
            if _current_mode == "replay":
                _current_mode = "live"
                _current_mission_name = None
                _replay_engine = None
                _sim_meta["mode"] = "live"
                _sim_meta["mission"] = None
                log.info("Exited mission replay mode — switched to live manual control mode.")

            _simulator.set_manual_control(throttle=throttle, injected_fault=injected_fault)
            _control_revision += 1
            _simulator.control_revision = _control_revision
            log.info("Manual simulator control set: throttle=%.2f, fault=%s", throttle, injected_fault)
        else:
            _simulator.clear_manual_control()
            _control_revision += 1
            _simulator.control_revision = _control_revision
            log.info("Manual simulator control cleared.")

    return {
        "status": "success",
        "mode": _current_mode,
        "control_revision": _control_revision,
        "manual_override": _simulator.manual_override,
        "throttle": _simulator.manual_throttle,
        "injected_fault": _simulator.injected_fault,
    }


class MultiParameterControlRequest(BaseModel):
    manual_override: bool = True
    injected_fault: str = "NONE"
    override_flags: dict[str, bool] = Field(default_factory=dict)
    override_values: dict[str, float] = Field(default_factory=dict)


@app.post("/control/manual", tags=["Control"], summary="Set independent multi-parameter manual overrides")
async def set_multi_parameter_manual_control(req: MultiParameterControlRequest) -> dict:
    """Set independent manual parameter overrides."""
    global _current_mode, _current_mission_name, _replay_engine, _control_revision
    async with _sim_lock:
        if _simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not initialised.")
        if req.manual_override and _current_mode == "replay":
            _current_mode = "live"
            _current_mission_name = None
            _replay_engine = None
            _sim_meta["mode"] = "live"
            _sim_meta["mission"] = None
        
        try:
            validate_overrides(req.override_flags, req.override_values)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        allowed_faults = {"NONE", "OVERHEATING", "LUBRICATION_ISSUE", "EXHAUST_LEAK",
                          "FUEL_RESTRICTION", "FUEL_SYSTEM_RESTRICTION", "IMPROPER_INJECTION",
                          "IMPROPER_INGESTION", "MISFIRE", "CYLINDER_MISFIRE"}
        if req.injected_fault.upper().strip() not in allowed_faults:
            raise HTTPException(status_code=422, detail=f"Unsupported injected_fault: {req.injected_fault}")
        _simulator.set_multi_parameter_control(
            manual_override=req.manual_override,
            injected_fault=req.injected_fault,
            override_flags=req.override_flags,
            override_values=req.override_values,
        )
        _control_revision += 1
        _simulator.control_revision = _control_revision
    return {
        "status": "success",
        "manual_override": _simulator.manual_override,
        "injected_fault": _simulator.injected_fault,
        "override_flags": _simulator.override_flags,
        "override_values": _simulator.override_values,
        "control_revision": _control_revision,
    }


@app.get("/control/manual", tags=["Control"], summary="Get multi-parameter manual control status")
async def get_multi_parameter_manual_control() -> dict:
    async with _sim_lock:
        if _simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not initialised.")
        return {
            "manual_override": _simulator.manual_override,
            "injected_fault": _simulator.injected_fault,
            "override_flags": _simulator.override_flags,
            "override_values": _simulator.override_values,
            "control_revision": _control_revision,
        }


@app.post("/simulator/reset_control", tags=["Control"], summary="Clear manual simulator controls")
async def reset_simulator_control() -> dict:
    """Clear manual simulator controls and return to automatic flight simulation."""
    global _control_revision
    async with _sim_lock:
        if _simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not initialised.")
        _simulator.clear_manual_control()
        _control_revision += 1
        _simulator.control_revision = _control_revision

    log.info("Manual simulator control reset.")
    return {"status": "success", "message": "Manual controls cleared. Automatic simulation restored.",
            "control_revision": _control_revision}


class EnvironmentControlPayload(BaseModel):
    profile: str = Field(default="ISA_STANDARD", description="Atmospheric profile (ISA_STANDARD, HIGH_HEAT_DESERT, ARCTIC_FREEZE, HIGH_ALTITUDE_THIN_AIR, LIVE_STORM_TURBULENCE)")
    altitude: float = Field(default=0.0, ge=0.0, le=50000.0, description="Flight altitude in feet MSL")
    ambient_temp: float = Field(default=25.0, ge=-50.0, le=60.0, description="Ambient temperature in °C")


@app.post("/control/environment", tags=["Control"], summary="Set atmospheric environment condition profile and altitude")
async def set_environment_control(req: EnvironmentControlPayload) -> dict:
    """Dynamically set atmospheric environmental conditions and flight altitude."""
    global _control_revision
    async with _sim_lock:
        if _simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not initialised.")
        try:
            _simulator.set_atmospheric_profile(
                profile=req.profile,
                altitude=req.altitude,
                ambient_temp=req.ambient_temp,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        _control_revision += 1
        _simulator.control_revision = _control_revision
    return {
        "status": "success",
        "atmospheric_profile": _simulator.atmospheric_profile,
        "altitude": _simulator.altitude,
        "ambient_temp": _simulator.ambient_temp,
        "baro_pressure": _simulator.baro_pressure,
        "air_density_ratio": _simulator.air_density_ratio,
        "control_revision": _control_revision,
    }


@app.get("/control/environment", tags=["Control"], summary="Get current atmospheric environment condition state")
async def get_environment_control() -> dict:
    async with _sim_lock:
        if _simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not initialised.")
        return {
            "atmospheric_profile": _simulator.atmospheric_profile,
            "altitude": _simulator.altitude,
            "ambient_temp": _simulator.ambient_temp,
            "baro_pressure": _simulator.baro_pressure,
            "air_density_ratio": _simulator.air_density_ratio,
            "control_revision": _control_revision,
        }


# ---------------------------------------------------------------------------
# Replay Endpoints
# ---------------------------------------------------------------------------

@app.get("/missions", tags=["Replay"], summary="List available mission recordings")
async def get_missions() -> dict:
    """Returns a list of all recorded mission CSV files available in data/missions/."""
    return {"missions": list_available_missions()}


@app.get("/replay/status", tags=["Replay"], summary="Get current replay status")
async def replay_status() -> dict:
    """Returns current system mode (live or replay) and mission progress."""
    return {
        "mode": _current_mode,
        "mission": _current_mission_name,
        "running": _sim_meta.get("running", False),
        "progress_pct": _sim_meta.get("replay_progress_pct", 0.0),
    }


@app.post("/replay/start", tags=["Replay"], summary="Start mission replay")
async def start_replay(
    mission: str = Query(default="mission_001", description="Mission recording name (e.g. mission_001)"),
) -> dict:
    """
    Start replaying a recorded mission CSV through the Digital Twin Core.
    """
    global _current_mode, _current_mission_name, _replay_engine, _twin, _control_revision
    async with _sim_lock:
        try:
            _replay_engine = MissionReplayEngine(mission)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load mission: {e}")

        # Re-initialize twin core for clean mission evaluation baseline
        _twin.reset()
        _control_revision += 1
        if _simulator is not None:
            _simulator.control_revision = _control_revision
        _current_mode = "replay"
        _current_mission_name = mission
        _sim_meta["mode"] = "replay"
        _sim_meta["mission"] = mission
        _sim_meta["running"] = True
        _sim_meta["replay_progress_pct"] = 0.0
        log.info("Started mission replay for '%s'", mission)

    return {"status": "started", "mode": "replay", "mission": mission}


@app.post("/replay/stop", tags=["Replay"], summary="Stop mission replay")
async def stop_replay() -> dict:
    """
    Stop active mission replay and return the Digital Twin to live simulation mode.
    """
    global _current_mode, _current_mission_name, _replay_engine, _simulator, _twin, _control_revision
    async with _sim_lock:
        _current_mode = "live"
        _current_mission_name = None
        _replay_engine = None
        _simulator.reset()
        _twin.reset()
        _control_revision += 1
        _simulator.control_revision = _control_revision
        _sim_meta["mode"] = "live"
        _sim_meta["mission"] = None
        _sim_meta["running"] = True
        _sim_meta["replay_progress_pct"] = 0.0
        log.info("Stopped mission replay. Switched to live simulation mode.")

    return {"status": "stopped", "mode": "live"}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def websocket_telemetry(
    ws: WebSocket,
    interval: float = Query(default=0.0, ge=0.0, le=10.0,
                            description="Optional additional delay between frames (seconds)"),
):
    """
    WebSocket push stream — delivers one :class:`TwinState` JSON frame per
    simulation tick (every ``dt`` seconds ≈ 0.5 s by default).

    Query parameters
    ----------------
    interval : float
        Extra sleep added between frames on the *client* side of this
        handler.  Set to 0 (default) for maximum update rate.
        Useful for lower-bandwidth connections or slow dashboards.

    Protocol
    --------
    - Client connects.
    - Server immediately sends the latest state (if available) so the
      client can initialise its display without waiting for the next tick.
    - Server then pushes new states on every simulation tick via
      :meth:`_ConnectionManager.broadcast`.
    - Client can send any text frame as a ping; server echoes ``{"pong": true}``.
    - Connection closes normally when the client disconnects.
    """
    await manager.connect(ws)

    try:
        # ── Prime the client with the latest known state ──────────────────
        if _latest_state is not None:
            await ws.send_json(_latest_state)

        # ── Keep alive: echo any client pings, wait for disconnect ────────
        while True:
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=_SIM_DT + interval + 1.0)
                # Client sent a message — treat as a ping
                await ws.send_json({"pong": True, "received": text[:64]})
            except asyncio.TimeoutError:
                # No client message this cycle — that's fine, keep waiting
                pass

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected cleanly.")
    except Exception as exc:
        log.warning("WebSocket error: %s", exc)
    finally:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------
@app.get("/api/parameters", tags=["Kinetic pipeline"])
async def parameters():
    return {"schema_version": 1, "parameters": parameter_catalog()}


@app.get("/api/pipeline", tags=["Kinetic pipeline"])
async def pipeline_assessment():
    async with _sim_lock:
        payload = dict(_latest_state or {})
    return {"status": payload.get("pipeline_status", "waiting"),
            "session_id": payload.get("session_id"),
            "control_revision": payload.get("control_revision"),
            "assessment_control_revision": payload.get("assessment_control_revision"),
            "control_pending": payload.get("control_pending", True),
            "assessment": payload.get("pipeline", {}),
            "model_status": payload.get("model_status", {}),
            "unsupported_metrics": payload.get("unsupported_metrics", {})}

def _dashboard_timestamps(value):
    """Preserve nanosecond integers beyond JavaScript's safe integer range."""
    if isinstance(value, dict):
        return {key: str(item) if key.endswith("_ns") and isinstance(item, int)
                else _dashboard_timestamps(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dashboard_timestamps(item) for item in value]
    return value


# Same-origin API used exclusively by the React console.
@app.get("/api/dashboard/state", tags=["React dashboard"])
async def dashboard_state():
    async with _sim_lock:
        telemetry = dict(_latest_state or {})
        revision = _control_revision
        session = _twin.session_id if _twin else None
        client = _twin.redis if _twin else None
        controls = {
            "manual_override": _simulator.manual_override if _simulator else False,
            "injected_fault": _simulator.injected_fault if _simulator else "NONE",
            "override_flags": dict(_simulator.override_flags) if _simulator else {},
            "override_values": dict(_simulator.override_values) if _simulator else {},
        }
        environment = {"atmospheric_profile": _simulator.atmospheric_profile,
                       "altitude": _simulator.altitude, "ambient_temp": _simulator.ambient_temp} if _simulator else {}
        age_ms = (time.monotonic() - _latest_state_received_at) * 1000 if _latest_state_received_at else None
        simulation = dict(_sim_meta)
    async def latest_record(stream):
        if client is None:
            return {}
        try:
            rows = await client.xrevrange(stream, count=1)
            if not rows:
                return {}
            item = json.loads(rows[0][1]["payload"])
            context = item.get("context") or {}
            stamp = int(item.get("timestamp_ns") or item.get("window_end_ns") or 0)
            if (item.get("session_id", context.get("session_id")) != session
                    or time.time_ns() - stamp > 15_000_000_000):
                return {}
            return item
        except Exception as err:
            log.warning("xrevrange stream %s error: %s", stream, err)
            return {}
    try:
        layer6, window = await asyncio.gather(latest_record("engine:layer6:assessments"),
                                              latest_record("engine:features:windows"))
    except Exception as exc:
        log.warning("Gather error: %s", exc)
        layer6, window = {}, {}
    window_summary = {key: window.get(key) for key in (
        "window_start_ns", "window_end_ns", "window_length_s", "data_quality_score",
        "feature_schema_version", "feature_schema_hash", "truncated", "cold_start")}
    window_summary["feature_count"] = len(window.get("features", {})) if window else None
    return _dashboard_timestamps({"telemetry": telemetry, "telemetry_age_ms": age_ms, "control_revision": revision,
            "simulation": simulation, "controls": controls, "environment": environment,
            "parameters": parameter_catalog(), "missions": list_available_missions(),
            "layer6": layer6, "window": window_summary})


class MaintenanceAdvisoryRequest(BaseModel):
    edge_ml: dict[str, Any]
    risk_rul: dict[str, Any]


@app.post("/api/dashboard/maintenance-advisory", tags=["React dashboard"])
async def dashboard_maintenance_advisory(req: MaintenanceAdvisoryRequest):
    """Generate from the exact results displayed when the button was pressed."""
    if not req.edge_ml.get("fault_class"):
        raise HTTPException(status_code=422, detail="No Edge ML fault classification is available.")
    try:
        return await asyncio.to_thread(draft_with_groq, req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.warning("Maintenance advisory request failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Unable to generate maintenance advisory from Groq. Please try again.") from exc


@app.post("/api/dashboard/controls/manual", tags=["React dashboard"])
async def dashboard_manual(req: MultiParameterControlRequest):
    return await set_multi_parameter_manual_control(req)


@app.post("/api/dashboard/controls/environment", tags=["React dashboard"])
async def dashboard_environment(req: EnvironmentControlPayload):
    return await set_environment_control(req)


@app.post("/api/dashboard/reset", tags=["React dashboard"])
async def dashboard_reset():
    result = await reset_simulation()
    return {**result, "control_revision": _control_revision}


class DashboardReplayRequest(BaseModel):
    mission: str


@app.post("/api/dashboard/replay/start", tags=["React dashboard"])
async def dashboard_replay_start(req: DashboardReplayRequest):
    if req.mission not in list_available_missions():
        raise HTTPException(status_code=422, detail="Choose an available mission recording.")
    result = await start_replay(req.mission)
    return {**result, "control_revision": _control_revision}


@app.post("/api/dashboard/replay/stop", tags=["React dashboard"])
async def dashboard_replay_stop():
    result = await stop_replay()
    return {**result, "control_revision": _control_revision}


_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="react-dashboard")
