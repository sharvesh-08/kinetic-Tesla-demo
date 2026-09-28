# MALE UAV engine pipeline — presentation outline

## 1. Purpose and scope

- Demonstrate a layered simulator/inference integration and interactive fault-injection workflow.
- Clearly label simulator-injected values and synthetic model data.
- State that the current checked-in LightGBM/Isolation Forest artifact and RUL prior are not decision-validated.

## 2. Seven-layer flow

1. Sensor simulator or asynchronous ECU events and Layer 1 validation.
2. Layer 2 common-timestamp synchronization with per-channel validity, source, age and rate.
3. Layer 3 PINN expected-state predictions and out-of-domain handling.
4. Layer 4 measurement residuals, update state and uncertainty.
5. Layer 5 five-second, 100 Hz feature windows with channel availability.
6. Layer 6 LightGBM and Isolation Forest under a versioned feature contract.
7. Layer 7 risk fusion, redline response, guarded RUL, and dashboard publishing.

## 3. Dashboard data contract

- Raw sensor values come from the valid current Layer 2 frame only.
- Expected values and residual vectors are separate from measured values.
- Manual controls inject simulated sensors, record provenance, and advance control revision.
- Dashboard displays pending while it waits for a matching Layer 7 assessment.
- Invalid or stale channel readings display unavailable.

## 4. Current readiness and next evidence

- Simulator rate: 2 Hz; Layer 2's 100 Hz grid holds samples between updates, so grid rate is not sensor rate.
- Registry: 672 features; Isolation Forest registered subset: 222 features.
- Classifier inference is shown for the current synthetic artifact and low-coverage windows, with an explicit warning; those results are excluded from Layer 7 risk fusion.
- Fuel-flow physics needs configured `FUEL_DENSITY_KG_L`.
- The provisional RUL estimate is visible, but remains unvalidated for maintenance decisions until its prior is validated against maintenance/fleet data.
- Decision readiness requires independently verified healthy data, representative fault labels, sortie-separated evaluation, and domain review.
