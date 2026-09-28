# Kinetic Tesla Demo — MALE UAV Digital Twin

A local demonstration dashboard for an aero-piston UAV engine. The project combines simulated engine telemetry, a Redis-connected processing pipeline, physics predictions, residual tracking, machine learning diagnostics, risk and remaining useful life (RUL) assessments, and Groq-generated maintenance advisories.

## Dashboard

The React console is served by FastAPI at **http://127.0.0.1:8000** and includes:

- **Overview / Digital Twin:** interactive Three.js engine viewer, synchronized engine readings and operating controls.
- **Diagnostics & Prediction:** physics/PINN and EKF residual table, LightGBM fault class and confidence, and Isolation Forest anomaly score.
- **Risk & RUL:** engine health state, subsystem risk, remaining useful life and maintenance advisory.
- **Trends & Telemetry:** live parameter charts and sensor histories.
- **Mission Replay:** replay recorded missions or return to the live simulator.

The control sidebar supports environment profiles, fault injection and manual sensor overrides. Edge ML and Risk/RUL displays capture the latest available data every 10 seconds; telemetry, charts and the engine viewer continue their normal updates.

## Requirements

- Python with the dependencies in `requirements.txt`.
- Node.js 20.19+ or a compatible newer release, and npm.
- Redis 5 or newer for the processing pipeline.
- Internet access and a Groq API key for maintenance advisory generation.

The bundled machine learning artifacts require compatible versions of their serialization dependencies, particularly scikit-learn, LightGBM and joblib.

## Local setup

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
npm ci --prefix frontend
cp .env.example .env
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Edit `.env` and add your own Groq key:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Keep `.env` private. It is excluded from version control. The advisory module also accepts these settings from the process environment.

## Start the application

```bash
python3 run.py
```

This launcher builds the frontend, checks for an available dashboard port, starts Redis on the configured local port if necessary, starts Layers 2–7 and the FastAPI backend, and opens the dashboard. Its default Redis URL is `redis://127.0.0.1:6380/15`.

If Redis is installed but not on your PATH, start it separately and configure the launcher:

```bash
redis-server --port 6380
```

Then, in another terminal:

```bash
export REDIS_URL=redis://127.0.0.1:6380/15
python3 run.py
```

Press **Ctrl+C** in the launcher terminal to stop the services it started. Stop an older dashboard process before restarting after Python changes.

Alternative launchers `run_demo.py` and `start_male_uav.py` build the same dashboard and expect Redis to be running; configure `REDIS_URL` when using a non-default Redis port.

## Processing pipeline

| Layer | Responsibility | Location |
| --- | --- | --- |
| 1 | Simulator telemetry, channel definitions, manual control adaptation | `app/simulator.py`, `layer1/` |
| 2 | Sensor synchronization, channel validity and common frames | `layer2/` |
| 3 | Physics/PINN predictions and model-domain handling | `layer3/`, `layer 3/` |
| 4 | EKF processing, measurement residuals and sensor-bias evidence | `layer4/` |
| 5 | Fixed feature windows and registered feature schemas | `layer5/` |
| 6 | LightGBM fault heads, probability calibration, Isolation Forest and SHAP | `layer6/` |
| 7 | Risk fusion, health state, trajectory/Bayesian RUL and dashboard publication | `layer7/` |

Redis streams connect the processing layers. FastAPI adapts their results into the dashboard API. The running launcher selects `configs/feature_registry_v4.yaml` with `models/layer6-male-uav-v4-synthetic-dev.joblib`; custom artifacts must use a matching feature registry.

## Model and display behavior

The default Layer 6 artifact uses synthetic development data. Its predictions remain unverified. Layer 7 provisionally considers fault probabilities at or above 60% and issues at least an advisory for those predictions. Health states also consider anomaly scores, physical residuals and redline evidence. RUL depends on the configured model/prior and available degradation evidence; fault-class confidence alone does not establish time to failure.

Isolation Forest scores use a smooth normalization of the stored healthy-score anchors rather than hard-clipping every out-of-range reading to exactly 1. The displayed score is an anomaly indicator, not a probability of engine failure.

When physics-table inputs are missing, `frontend/src/physicsDisplay.js` supplies dynamic display values based on simulator equations and time. Available pipeline readings take priority. These fallback values are confined to the table and do not feed the EKF, feature windows, ML inference or risk fusion.

This repository is a simulation/demo. Bundled models, thresholds, synthetic data and RUL priors do not establish validated aircraft maintenance or flight decisions.

## Maintenance advisory

Press **Generate Maintenance Advisory** in Risk & RUL. The frontend sends the currently displayed fault class, confidence, anomaly score, health state, risk values, RUL and action advisory to `/api/dashboard/maintenance-advisory`.

`app/maintenance_advisory.py` reads the Groq configuration, defines the system prompt and places those dashboard results in the user prompt. Groq's final text is displayed in the existing advisory area. The API key stays on the server. Provider errors are returned as dashboard errors rather than replaced with a canned advisory.

## Repository contents

| Directory / file | Contents |
| --- | --- |
| `app/` | FastAPI, simulator, pipeline adapter, replay, physics helper and Groq advisory |
| `frontend/` | React/Vite console, styles, HTML enhancements and Three.js assets |
| `layer1/`–`layer7/` | Processing services, feature contracts, inference and risk logic |
| `layer 3/` | Physics model artifacts and normalization metadata |
| `configs/` | Feature registries, risk policy and runtime configuration |
| `models/` | Bundled inference and RUL artifacts |
| `data/` | Telemetry, mission recordings and development datasets |
| `fixtures/` | Reproducible development inputs and reports |
| `tests/` | Unit, service, pipeline and live-verification checks |
| `tools/` | Supporting utilities |
| `docs/`, `reports/` | Architecture material and project reports |
| `run.py` | All-in-one local launcher |
| `Dockerfile`, `docker-compose.yml` | Container build and Redis/backend services |

Existing per-layer README files are omitted from this demo repository; this document is its overall README. Local dependencies, build output, caches, private environment files and the redundant project ZIP are also omitted. Model files and browser assets required by the demonstration are included.

## Useful endpoints

- Dashboard: `http://127.0.0.1:8000/`
- API documentation: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Dashboard state: `http://127.0.0.1:8000/api/dashboard/state`
- Engine viewer: `http://127.0.0.1:8000/engine/index.html`

## Development and checks

```bash
npm run build --prefix frontend
npm test --prefix frontend
python3 -m unittest tests.test_layer7 tests.test_layer7_dashboard tests.test_maintenance_advisory
```

For broader checks:

```bash
python3 -m unittest discover -s tests -v
```

Some integration checks need a reachable Redis server, optional datasets or the appropriate model/feature-registry configuration. Frontend development is available through `npm run dev --prefix frontend` with the backend running separately.

## Docker

```bash
docker compose up --build
```

Compose starts Redis and the backend and publishes port 8000. To use maintenance advisory in a container, provide `GROQ_API_KEY` and optionally `GROQ_MODEL` to the backend service through your local Compose environment configuration; the checked-in Compose file does not include credentials.

## Troubleshooting

- **Missing `frontend/dist`:** run `npm ci --prefix frontend`, then restart the launcher.
- **Port 8000 already in use:** stop the earlier run before starting another instance.
- **Redis unavailable:** start Redis or set `REDIS_URL` to the correct reachable instance.
- **Feature schema mismatch:** configure `FEATURE_REGISTRY_PATH` and `LAYER6_ARTIFACT` as a matched pair.
- **Groq advisory fails:** check your private API key and that `GROQ_MODEL` is available to your account. Restart the backend after source changes.
- **Cached dashboard assets:** reload the browser after rebuilding the frontend.
