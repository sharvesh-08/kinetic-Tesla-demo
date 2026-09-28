# React dashboard migration

The Streamlit application and its old Three.js directory were removed after the React production build and live browser checks. `run_demo.py` now launches the React dashboard served by FastAPI at port 8000. `start_male_uav.py` builds React, then starts Layers 2–7 and the API. No process binds port 8501.

The engine viewer was migrated to `frontend/public/engine`. Its former Streamlit listener, hardcoded port-8000 WebSocket, and independent HTTP polling were removed. It accepts only typed messages from its same-origin parent. React owns one polling connection with a timeout and reconnect behavior; both viewer and tables consume that snapshot. The canvas stays mounted while telemetry changes.

The API facade under `/api/dashboard` uses the existing simulator control handlers and validation. It exposes the current Layer 6 record rather than relying only on Layer 7's reduced projection, and includes Layer 5's actual feature count/schema. It excludes stale records and foreign sessions. The frontend withholds old assessment revisions after controls change.

The Layer 2 frame is shown immediately below the viewer. Its nanosecond timestamps are strings, and invalid channels show unavailable. The physics comparison uses the measurement frame retained with the Layer 4 residual; its timestamp is displayed separately from the newest Layer 2 frame.

Advertised electrical and injection overrides previously passed validation but were absent from the simulator's override map. The simulator now initializes every catalog parameter and applies those overrides to its outgoing telemetry with provenance.

Validation includes frontend tests for bounded history, session resets, null outputs and control revision matching; backend tests for Layer 6 status, feature counts, stale-record rejection, timestamp precision and every advertised override. The production React page was tested with real Redis and Layers 1–7, including changing telemetry and controls.

Layer 6 displays inference from the bundled development artifact for compatible windows and labels its training scope; unverified inference is excluded from Layer 7 risk fusion. RUL is still unvalidated for decisions. Docker configuration was migrated to one dashboard/API service plus Redis; container execution requires Docker and is a separate deployment check.
