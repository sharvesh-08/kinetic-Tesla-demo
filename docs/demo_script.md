# MALE UAV pipeline demonstration

This script demonstrates integration behavior only. It does not claim operational fault accuracy or validated RUL.

## Start

1. Start Redis 5 or later.
2. Run `python start_male_uav.py`.
3. Open the React dashboard on port 8000 and wait for a fresh Layer 2 frame and matching Layer 7 assessment.

## Normal simulator stream

Show the active session, Layer 2 common timestamp, channel validity/source/age, Layer 3 model statuses, Layer 4 updated residuals, Layer 5 window schema/availability, Layer 6 LightGBM and Isolation Forest inference with its training scope and coverage warning, and Layer 7 status. The checked-in synthetic development artifact's inference is excluded from Layer 7 risk fusion. Show the provisional RUL simulation estimate with its unvalidated-prior warning; do not present it as maintenance guidance or describe unavailable health as nominal.

## Sensor override

Use the manual controls to inject a bounded RPM, EGT, or altitude sensor value. Confirm that:

- the dashboard raw reading matches the Layer 2 synchronized frame;
- provenance labels the override simulated and records the control revision;
- the dashboard marks the assessment pending until Layer 7 catches up to that revision;
- valid Layer 3/4 predictions and residuals carry matching timestamps and availability;
- invalid or out-of-domain channels remain unavailable instead of showing stale values.

Change to another control set and confirm omitted override flags clear. Use Reset Controls to restore simulation values.

## Atmosphere control

Select `ISA_STANDARD` and check its altitude and temperature. Then select high altitude and confirm both the selected profile and requested values appear in the Layer 2 frame and dashboard. API values are authoritative; the UI must not map a request to a different profile's altitude or temperature.

## Reading results

Layer 6 inference appears for compatible windows even when the artifact or sensor coverage is unverified; those predictions are excluded from Layer 7 risk fusion. Risk and maintenance recommendations are advisory until their models and thresholds receive domain validation. RUL hours and intervals are provisional until the prior is validated against relevant engine maintenance outcomes.
