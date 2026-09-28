# MALE UAV integration

The active engine path is Layer 1 validation → Layer 2 synchronization → Layer 3 PINN physics → Layer 4 residual estimation → Layer 5 feature windows → Layer 6 classifiers → Layer 7 risk/RUL and dashboard publishing. The simulator runs inside the MALE API process. Both `python run_demo.py` and `python start_male_uav.py` start the API, Redis-backed Layers 2–7, and React dashboard. `run_demo.py` waits for API simulation and dashboard readiness before opening the browser.

## Dashboard sensor source

Every raw engine sensor value rendered in the dashboard is taken from the current-session, fresh, valid Layer 2 synchronized frame. The adapter prefers `engine:synced:frames`; it can use the frame embedded in the Layer 7 assessment when the direct frame is unavailable. It never fills an absent sensor with simulator state, a PINN prediction, an EKF estimate, or a zero. Expected values and residuals remain separate. The dashboard shows per-channel source, age, rate, and validity metadata.

## Sampling and dependency behavior

The simulator currently emits at 2 Hz. Layer 2 maps asynchronous channel events to a 100 Hz master grid using held values between samples; held grid points are not new measurements. Layer 5 makes complete 5-second windows with 50% overlap, but availability features preserve how much real channel data exists. The active v4 registry contains 672 features. LightGBM consumes its registered feature vector; Isolation Forest consumes the 222-feature registered subset.

Layer 3 and Layer 4 validity propagate forward. PINN out-of-domain results and missing dependencies are unavailable, and an EKF residual is usable only with an updated measurement and valid prediction. Fuel-flow physics requires `FUEL_DENSITY_KG_L`; without it, fuel-flow residual availability is zero. Layer 6 runs inference for complete schema-compatible windows regardless of training scope and sensor coverage. Unverified training scope or sensor coverage below 0.05 keeps inference out of Layer 7 risk fusion.

## Interactive controls

`/api/dashboard/state` supplies slider limits and current controls. `/api/dashboard/controls/manual` accepts sensor overrides; a subsequent control request replaces the active override flags, and posting empty override flags with manual mode disabled clears overrides. `control_revision` increments for control, environment, reset, and replay changes and propagates through the frames/windows/assessments. The UI shows a pending state until the latest assessment revision matches. Provenance marks UI-injected readings as simulated. They exercise fault response paths; they do not cause the physical simulator equations to recompute all coupled engine channels.

Atmospheric controls honor the requested altitude and ambient temperature. Named profiles provide defaults only when the API request leaves altitude at 0 ft and temperature at 25 °C. `ISA_STANDARD` is the initial dashboard selection.

## Model and RUL status

The repository's current Layer 6 artifact is a synthetic development artifact with unverified nominal data. Layer 6 still exposes LightGBM and Isolation Forest inference for compatible windows, while marking it ineligible for Layer 7 risk fusion. No synthetic classifier retraining can establish engine-domain accuracy. A decision-capable artifact needs independently verified healthy maintenance sorties, representative labeled faults, sortie-separated calibration and holdout evaluation, plus domain review.

Layer 7's current RUL prior is also unvalidated for decisions. The dashboard displays its provisional simulation estimate and interval with an explicit warning; it is not suitable for maintenance decisions until validated against real fleet/maintenance outcomes.

## Local services

Run `npm ci --prefix frontend` once and start Redis 5 or later, then run `python run_demo.py` (or `python start_male_uav.py`). Set `REDIS_URL` to select the broker. Set `FUEL_DENSITY_KG_L` to enable calibrated fuel-flow physics. Set `FEATURE_REGISTRY_PATH` and `LAYER6_ARTIFACT` as a matched pair when loading a validated model artifact.
