# MALE UAV project documentation

The active dashboard is a React application served with FastAPI at **http://127.0.0.1:8000**. `python3 run_demo.py` builds React and starts the full seven-layer pipeline. The Streamlit dashboard and its launcher connections have been removed.

- [Setup, controls, endpoints and model limitations](README.md)
- [Seven-layer architecture and data provenance](docs/architecture.md)
- [Engine integration and model dependencies](docs/male_uav_integration.md)
- [React migration and verification](docs/react_dashboard.md)

Frontend source is in `frontend/src/`. The Three.js engine viewer and local vendor files are in `frontend/public/engine/`. The synchronized Layer 2 frame appears below the viewer. Physics and residuals, Layer 5/6 diagnostics, and Layer 7 risk/RUL follow below it.

The React client uses `/api/dashboard/state` and `/api/dashboard/controls/*`. Inputs change the simulator; displayed sensor measurements come from Layer 2. Manual controls carry revisions through the pipeline, and the dashboard withholds assessment outputs until revisions match. Compatible Layer 6 windows show inference with coverage/training warnings; the provisional RUL estimate is shown with its validation status.
