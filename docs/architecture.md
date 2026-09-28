# MALE UAV Engine Pipeline Architecture

This document describes the active MALE UAV pipeline started by `python run_demo.py` or `python start_male_uav.py`. The dashboard is a consumer of the pipeline outputs; it does not calculate a second sensor stream.

## Data flow

```text
Engine simulator / ECU events
  -> Layer 1 validation and event envelope
  -> Layer 2 per-channel timestamp synchronization
  -> Layer 3 PINN physics predictions
  -> Layer 4 federated residual estimation
  -> Layer 5 fixed 5 s windows and registered features
  -> Layer 6 LightGBM + Isolation Forest
  -> Layer 7 risk fusion, action and guarded RUL
  -> dashboard publisher -> FastAPI/WebSocket -> React dashboard
```

The dashboard's raw sensor values come exclusively from `engine:synced:frames` (Layer 2), or from the same frame embedded in a Layer 7 assessment when the direct frame is not available. A frame must match the active session, be fresh, and mark its channel valid. Invalid, absent, or stale channels display as unavailable. Layer 3 predictions and Layer 4 residuals remain separate fields and are never substituted for sensor readings.

## Layer contracts

| Layer | Active responsibility | Contract and current limits |
| --- | --- | --- |
| 1 — source and validation | The simulator in `app/main.py` emits sensor events into `layer1`; ECU adapters can submit asynchronous channel events. | Every event has a channel, value, timestamp, units, session and provenance. Manual overrides are simulated inputs, not measured engine data. |
| 2 — synchronization | `layer2` aligns asynchronous channels to a common master timestamp and publishes `engine:synced:frames`. | Per-channel validity, source, age and rate accompany values. The current simulator is 2 Hz; the 100 Hz grid uses held values between fresh samples, so grid rate is not measurement rate. |
| 3 — physics/PINNs | `layer3` infers healthy expected channel values and marks model/channel validity. | Out-of-domain or dependency-missing predictions are unavailable. Fuel-flow prediction requires configured `FUEL_DENSITY_KG_L`. |
| 4 — residual estimation | `layer4` updates per-channel residuals and uncertainty from predictions and measurements. | A residual is usable only when its measurement is updated and its physics prediction is valid. The dashboard uses the EKF prediction/residual from the same timestamp. |
| 5 — windowing | `layer5` assembles complete 5-second, 100 Hz windows with 50% overlap and emits named features. | The registry is authoritative for the active artifact. Availability features describe missing/held observations; windows are not treated as fully measured merely because they have 500 grid points. |
| 6 — classification | `layer6` runs LightGBM and Isolation Forest using the versioned feature schema. | The current artifact is `synthetic_nominal_unverified`; inference is displayed for compatible windows, including low-coverage windows, and marked ineligible for Layer 7 fusion until model provenance and sensor coverage qualify. The active registry contains 672 features; the Isolation Forest uses its registered 222-feature subset. |
| 7 — fusion and prognosis | `layer7` fuses classifier/anomaly/trend evidence, checks redlines and produces action and RUL metadata. | Withheld classifier results are not presented as predictions. RUL hours and intervals are exposed only if the configured prior is validated for decisions; current synthetic prior is not validated. |
| Dashboard | Layer 7 publisher combines assessment and synchronized Layer 2 frame. FastAPI serves the React dashboard and exposes the joined snapshot through `/api/dashboard/state`. | UI controls write simulator overrides and atmospheric settings. Control revision and simulated provenance follow the resulting frame/assessment; the dashboard marks assessments pending until revisions match. |

## Control behavior

`/control/manual` and `/simulator/control` apply explicit simulated overrides. A new request replaces the active override flags, so omitted channels return to simulation. `/simulator/reset_control` clears them. Atmosphere controls honor explicit altitude and ambient temperature; profiles provide defaults only when the request leaves the default atmosphere values. Every change increments `control_revision`.

Sensor overrides alter the emitted sensor reading while the simulator's engine-state equations continue independently. This is appropriate for testing sensor faults; it does not model the full physical response to a changed engine command. Provenance must be read before treating injected values as measured data.

## Running

Start Redis 5 or newer, then run `python run_demo.py` or `python start_male_uav.py`. The launcher starts the MALE API and Layers 2–7; Layer 1 runs in the API process to preserve a single simulator source. `REDIS_URL` selects Redis. Set `FUEL_DENSITY_KG_L` when fuel-flow physics is required. Configure `FEATURE_REGISTRY_PATH` and `LAYER6_ARTIFACT` together; an artifact and registry with different schemas must fail closed.

## Operational data requirements

Synthetic generated sorties are useful for integration checks but do not validate engine fault accuracy, operational risk thresholds, or RUL. A decision-capable Layer 6 artifact requires independently verified healthy nominal training data, representative labeled faults, sortie-separated calibration and holdout data, and domain review. A decision-capable RUL prior requires validated fleet/maintenance outcomes. Until these are available, show model outputs as unvalidated simulation results and exclude them from operational decisions.
