"""
Physics / Expected-Behaviour Model for UAV Aero-Piston Engine.

This module provides a lightweight empirical model that predicts what the
engine's key thermal and mechanical parameters *should* be under normal,
healthy operating conditions given the current throttle setting, shaft
speed (RPM), and ambient temperature.

Design notes
------------
The equations are intentionally aligned with the simulator's true-state
equations (before noise and fault injection), so that residuals produced
by the Digital Twin Core are near-zero when the engine is healthy and
deviate measurably when a sensor fault or physical degradation is present.

An ambient-temperature correction is applied on top of the base equations
to account for the effect of intake air density on combustion temperature
and heat-transfer rates — a real (though simplified) physical effect.

This is a demonstration model; it is NOT a certified aero-engine model.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Model coefficients  (kept in one place so they are easy to calibrate)
# ---------------------------------------------------------------------------

# --- Base engine map (must match simulator steady-state equations) ---
_EGT_THROTTLE_GAIN = 600.0  # °C per unit throttle
_EGT_RPM_GAIN = 0.05        # °C per RPM above idle reference
_RPM_IDLE_REF = 2000.0      # RPM idle reference (matching simulator)

_MAP_BASE = 15.0            # inHg — MAP at idle (vacuum)
_MAP_THROTTLE_GAIN = 14.0   # inHg increase up to WOT (~29.0 inHg)

_OIL_PRESS_BASE = 52.0      # PSI — nominal oil pressure at idle
_OIL_PRESS_RPM_GAIN = 0.008 # PSI per excess RPM
_OIL_PRESS_TEMP_COUPLING = -0.15 # PSI reduction per °C oil temp rise above base
_OIL_BASE = 80.0            # °C — reference oil temperature for pressure coupling

_FUEL_FLOW_BASE = 3.2       # L/h — idle fuel burn
_FUEL_FLOW_THROTTLE_GAIN = 12.5 # L/h per unit throttle
_FUEL_FLOW_RPM_GAIN = 0.001 # L/h per excess RPM

_IAT_BASE = 25.0            # °C — base intake air temp
_IAT_THROTTLE_GAIN = 8.0    # °C manifold rise under full throttle

_VIB_BASE = 0.35            # g RMS — baseline smooth operation vibration
_VIB_RPM_GAIN = 0.0003      # g per excess RPM above idle

# --- Environmental coefficients ---
_AMBIENT_REF = 25.0         # °C — ISA standard-day reference temperature
_IAT_AMBIENT_GAIN = 0.95    # additional °C IAT per °C above ambient ref

# --- Physical sanity limits (output is clipped to these) ---
_EGT_MAX = 1200.0                     # °C
_CHT_MAX = 600.0                      # °C
_OIL_MAX = 200.0                      # °C
_MAP_MAX = 45.0                       # inHg
_OIL_PRESS_MAX = 120.0                # PSI
_FUEL_FLOW_MAX = 40.0                 # L/h
_IAT_MIN, _IAT_MAX = -20.0,  110.0   # °C
_VIB_MIN, _VIB_MAX =  0.1,   15.0    # g


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ExpectedEngineState:
    """
    Predicted (expected) engine parameter values under healthy operation.

    Attributes
    ----------
    egt_expected : float
    cht_expected : float
    oil_temp_expected : float
    map_expected : float
    oil_press_expected : float
    fuel_flow_expected : float
    iat_expected : float
    vibration_expected : float
    rpm_input : float
    throttle_input : float
    ambient_temp_input : float
    """
    egt_expected: float
    cht_expected: float
    oil_temp_expected: float
    map_expected: float
    oil_press_expected: float
    fuel_flow_expected: float
    iat_expected: float
    vibration_expected: float
    # ── inputs echoed for traceability ────────────────────────────────────
    rpm_input: float
    throttle_input: float
    ambient_temp_input: float
    altitude_input: float = 0.0
    baro_pressure: float = 29.92
    air_density_ratio: float = 1.0

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "egt_expected": round(self.egt_expected, 4),
            "cht_expected": round(self.cht_expected, 4),
            "oil_temp_expected": round(self.oil_temp_expected, 4),
            "map_expected": round(self.map_expected, 4),
            "oil_press_expected": round(self.oil_press_expected, 4),
            "fuel_flow_expected": round(self.fuel_flow_expected, 4),
            "iat_expected": round(self.iat_expected, 4),
            "vibration_expected": round(self.vibration_expected, 4),
            "rpm_input": round(self.rpm_input, 2),
            "throttle_input": round(self.throttle_input, 4),
            "ambient_temp_input": round(self.ambient_temp_input, 2),
            "altitude_input": round(self.altitude_input, 1),
            "baro_pressure": round(self.baro_pressure, 2),
            "air_density_ratio": round(self.air_density_ratio, 4),
        }


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_expected(
    rpm: float,
    throttle: float,
    ambient_temp: float = 25.0,
    altitude: float = 0.0,
) -> ExpectedEngineState:
    """
    Predict expected engine parameter values under healthy operation.

    ``altitude`` is in metres. The equations mirror the simulator's steady
    state model; temperatures converge to ambient after shutdown, while
    shaft-driven pressure, fuel flow, and vibration fall to zero.
    """
    # ── Input sanitisation ────────────────────────────────────────────────
    rpm = max(0.0, float(rpm))
    throttle = float(max(0.0, min(1.0, throttle)))
    ambient_temp = float(ambient_temp)
    altitude = max(0.0, float(altitude))
    
    # Calculate ISA Barometric Altitude Pressure (inHg) and Air Density Ratio (sigma)
    # ISA barometric relation with altitude in metres (simulator control/API
    # altitude is feet and is converted at the twin boundary).
    p_baro = max(8.0, min(31.0, 29.92 * ((1.0 - 2.25577e-5 * altitude) ** 5.2559)))
    sigma = (p_baro / 29.92) * (298.15 / (273.15 + ambient_temp))

    # Engine load rises with shaft speed up to the rated idle reference. The
    # temperatures then follow the combustion/heat-transfer coupling used by
    # EngineSimulator, rather than using unrelated nominal offsets.
    running = rpm > 0.0
    load = min(1.0, rpm / _RPM_IDLE_REF) if running else 0.0
    egt_expected = ambient_temp
    cht_expected = ambient_temp
    oil_temp_expected = ambient_temp
    if running:
        egt_expected = ambient_temp + load * (
            375.0 + _EGT_THROTTLE_GAIN * throttle
            + _EGT_RPM_GAIN * (rpm - _RPM_IDLE_REF) + (1.0 - sigma) * 55.0
        )
        cht_expected = ambient_temp + load * (
            95.0 + 0.10 * (egt_expected - 400.0)
        )
        oil_temp_expected = ambient_temp + load * (
            55.0 + 0.06 * (egt_expected - 400.0)
        )
    egt_expected = float(_clip(egt_expected, ambient_temp, _EGT_MAX))
    cht_expected = float(_clip(cht_expected, ambient_temp, _CHT_MAX))
    oil_temp_expected = float(_clip(oil_temp_expected, ambient_temp, _OIL_MAX))

    # MAP is absolute manifold pressure: throttle moves it toward local
    # barometric pressure, and altitude caps it. Temperature changes charge
    # density and fuel demand, not static pressure at a fixed throttle.
    map_expected = min(p_baro, _MAP_BASE + _MAP_THROTTLE_GAIN * throttle) if running else p_baro
    map_expected = float(max(5.0, min(_MAP_MAX, map_expected)))

    if running:
        oil_press_expected = (
            _OIL_PRESS_BASE + _OIL_PRESS_RPM_GAIN * (rpm - _RPM_IDLE_REF)
            + _OIL_PRESS_TEMP_COUPLING * (oil_temp_expected - _OIL_BASE)
        )
        fuel_flow_expected = (
            _FUEL_FLOW_BASE + _FUEL_FLOW_THROTTLE_GAIN * throttle
            + _FUEL_FLOW_RPM_GAIN * (rpm - _RPM_IDLE_REF)
        ) * max(0.4, sigma)
        vibration_expected = _VIB_BASE + _VIB_RPM_GAIN * (rpm - _RPM_IDLE_REF)
    else:
        oil_press_expected = fuel_flow_expected = vibration_expected = 0.0
    oil_press_expected = float(max(0.0, min(_OIL_PRESS_MAX, oil_press_expected)))
    fuel_flow_expected = float(max(0.0, min(_FUEL_FLOW_MAX, fuel_flow_expected)))
    vibration_expected = float(max(0.0, min(_VIB_MAX, vibration_expected)))

    # Intake temperature follows ambient air plus load-related manifold heat.
    iat_expected = (25.0 + _IAT_THROTTLE_GAIN * throttle
                    + _IAT_AMBIENT_GAIN * (ambient_temp - _AMBIENT_REF)) if running else ambient_temp
    iat_expected = float(max(_IAT_MIN, min(_IAT_MAX, iat_expected)))

    return ExpectedEngineState(
        egt_expected=egt_expected,
        cht_expected=cht_expected,
        oil_temp_expected=oil_temp_expected,
        map_expected=map_expected,
        oil_press_expected=oil_press_expected,
        fuel_flow_expected=fuel_flow_expected,
        iat_expected=iat_expected,
        vibration_expected=vibration_expected,
        rpm_input=rpm,
        throttle_input=throttle,
        ambient_temp_input=ambient_temp,
        altitude_input=altitude,
        baro_pressure=p_baro,
        air_density_ratio=sigma,
    )


def _clip(value: float, lower: float, upper: float) -> float:
    """Small scalar clamp kept dependency-free for this physics module."""
    return max(lower, min(upper, value))


# ---------------------------------------------------------------------------
# EnginePhysicsModel — thin OOP wrapper (for DI / future extension)
# ---------------------------------------------------------------------------

class EnginePhysicsModel:
    """
    Object-oriented wrapper around :func:`predict_expected`.

    Provides a stable interface for the Digital Twin Core to call, and a
    natural extension point for future model updates (e.g. degradation
    maps, altitude corrections, fuel-type variants).

    Parameters
    ----------
    ambient_temp : float
        Default ambient temperature in °C used when no override is supplied.
    """

    def __init__(self, ambient_temp: float = 25.0) -> None:
        self.ambient_temp = ambient_temp

    def predict(
        self,
        rpm: float,
        throttle: float,
        ambient_temp: float | None = None,
        altitude: float = 0.0,
    ) -> ExpectedEngineState:
        """
        Return expected engine state for the given operating point.

        Parameters
        ----------
        rpm : float
            Engine speed (RPM).
        throttle : float
            Throttle position [0.0 – 1.0].
        ambient_temp : float | None
            Ambient temperature override (°C). Uses instance default when
            ``None``.
        altitude : float
            Altitude in feet above mean sea level.
        """
        t_amb = self.ambient_temp if ambient_temp is None else float(ambient_temp)
        return predict_expected(rpm=rpm, throttle=throttle, ambient_temp=t_amb, altitude=altitude)


# ---------------------------------------------------------------------------
# Quick self-test  (run: python -m app.physics_model)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("EnginePhysicsModel — sweep over throttle at ISA standard day\n")
    model = EnginePhysicsModel(ambient_temp=25.0)

    print(f"{'throttle':>10} {'rpm':>8} {'EGT_exp':>10} {'CHT_exp':>10} {'OIL_exp':>10}")
    print("-" * 52)

    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rpm_nominal = 2000 + 2500 * thr
        s = model.predict(rpm=rpm_nominal, throttle=thr)
        print(
            f"{thr:>10.2f} {rpm_nominal:>8.0f} "
            f"{s.egt_expected:>10.1f} {s.cht_expected:>10.1f} {s.oil_temp_expected:>10.1f}"
        )

    print("\nAmbient-temperature effect on EGT (throttle=0.6, rpm=3500)\n")
    print(f"{'ambient °C':>12} {'EGT_exp':>10} {'CHT_exp':>10}")
    print("-" * 36)
    for t_amb in [0, 10, 20, 25, 30, 40, 50]:
        s = predict_expected(rpm=3500, throttle=0.6, ambient_temp=t_amb)
        print(f"{t_amb:>12.0f} {s.egt_expected:>10.1f} {s.cht_expected:>10.1f}")
