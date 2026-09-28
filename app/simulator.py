"""
UAV Aero-Piston Engine & Sensor Telemetry Simulator.

Provides a continuous time-series simulation of engine operating parameters
with realistic physics-inspired state equations, Gaussian sensor noise, and
a configurable EGT sensor fault injection profile.
"""

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants / Defaults
# ---------------------------------------------------------------------------

# Throttle profile
_THROTTLE_CENTRE = 0.4
_THROTTLE_AMPLITUDE = 0.3
_THROTTLE_FREQ = 0.05          # rad/s  (slow mission-profile sweep)
_THROTTLE_NOISE_STD = 0.01     # small operational variation
_THROTTLE_MIN = 0.2
_THROTTLE_MAX = 0.9

# RPM model
_RPM_IDLE = 2000               # RPM at zero throttle reference
_RPM_GAIN = 2500               # RPM increase per unit throttle
_RPM_NOISE_STD = 20.0          # sensor / shaft variation (RPM)

# EGT model — first-order linear function of throttle and RPM
_EGT_OFFSET = 400.0            # base EGT at idle (°C)
_EGT_THROTTLE_GAIN = 600.0     # °C per throttle unit
_EGT_RPM_GAIN = 0.05           # °C per excess RPM above 2000
_EGT_NOISE_STD = 5.0           # sensor noise (°C)

# CHT model — driven by EGT thermal coupling
_CHT_OFFSET = 120.0            # base CHT (°C)
_CHT_EGT_COUPLING = 0.10       # °C CHT per °C EGT above 400 (nominal cruise ~140°C)
_CHT_NOISE_STD = 3.0           # sensor noise (°C)

# Oil temperature model
_OIL_TEMP_OFFSET = 80.0        # base oil temp (°C)
_OIL_TEMP_EGT_COUPLING = 0.06  # °C oil per °C EGT above 400 (nominal cruise ~92°C)
_OIL_TEMP_NOISE_STD = 2.0      # sensor noise (°C)

# MAP model (Manifold Absolute Pressure in inHg)
_MAP_OFFSET = 15.0             # base idle MAP (inHg vacuum)
_MAP_THROTTLE_GAIN = 14.0      # inHg gain up to wide open throttle
_MAP_NOISE_STD = 0.2           # sensor noise (inHg)

# Oil Pressure model (PSI)
_OIL_PRESS_OFFSET = 52.0       # base oil pressure at idle (PSI)
_OIL_PRESS_RPM_GAIN = 0.008    # PSI gain per RPM
_OIL_PRESS_TEMP_COUPLING = -0.15 # PSI reduction per °C oil temp rise above 80
_OIL_PRESS_NOISE_STD = 1.2     # sensor noise (PSI)

# Fuel Flow model (L/h)
_FUEL_FLOW_OFFSET = 3.2        # base idle fuel flow (L/h)
_FUEL_FLOW_THROTTLE_GAIN = 12.5 # L/h gain per throttle unit
_FUEL_FLOW_RPM_GAIN = 0.001    # L/h per excess RPM
_FUEL_FLOW_NOISE_STD = 0.15    # sensor noise (L/h)

# Intake Air Temp model (IAT in °C)
_IAT_OFFSET = 25.0             # base IAT (°C)
_IAT_THROTTLE_GAIN = 8.0       # °C manifold heating under load
_IAT_NOISE_STD = 0.8           # sensor noise (°C)

# Vibration model (RMS g)
_VIB_OFFSET = 0.35             # base vibration level (g)
_VIB_RPM_GAIN = 0.0003         # g increase per excess RPM
_VIB_NOISE_STD = 0.04          # sensor noise (g)

# Fault injection
_FAULT_TRIGGER_TIME = 60.0     # seconds — EGT bias activates after this time
_EGT_FAULT_BIAS = 40.0         # °C positive bias introduced by sensor fault
def _barometric_pressure_inhg(altitude_ft: float) -> float:
    """ISA pressure for the simulator's altitude input, which is in feet."""
    altitude_ft = max(0.0, float(altitude_ft))
    pressure = 29.92 * ((1.0 - 6.8755e-6 * altitude_ft) ** 5.2559)
    return max(8.0, min(31.0, pressure))


class EngineSimulator:
    """
    Simulates a UAV aero-piston engine telemetry stream.

    The simulation advances in fixed time-steps (dt). On each call to
    :meth:`step` the internal state is updated and a snapshot of all
    measured parameters is returned as a dictionary.
    """

    def __init__(
        self,
        dt: float = 0.5,
        seed: int | None = None,
        enable_fault: bool = False,
        mission_profile: Any = None,
        ambient_temp: float = 25.0,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        # ── Time ──────────────────────────────────────────────────────────
        self.dt: float = dt
        self.t: float = 0.0
        self.enable_fault: bool = enable_fault
        self.ambient_temp: float = ambient_temp
        self.atmospheric_profile: str = "ISA_STANDARD"
        self.altitude: float = 0.0
        self.baro_pressure: float = 29.92
        self.air_density_ratio: float = 1.0
        self.control_revision: int = 0

        # ── Mission profile ───────────────────────────────────────────────
        self._mission_profile_fn = None
        if mission_profile is not None:
            if callable(mission_profile):
                self._mission_profile_fn = mission_profile
            elif isinstance(mission_profile, str):
                from app.mission_profiles import get_mission_profile
                self._mission_profile_fn = get_mission_profile(mission_profile)

        # ── Engine state (initialised at plausible idle values) ───────────
        self.throttle: float = _THROTTLE_CENTRE
        self.rpm: float = _RPM_IDLE + _RPM_GAIN * _THROTTLE_CENTRE
        self.egt: float = _EGT_OFFSET                # measured EGT (°C)
        self.cht: float = _CHT_OFFSET                # measured CHT (°C)
        self.oil_temp: float = _OIL_TEMP_OFFSET      # measured oil temp (°C)
        self.map: float = _MAP_OFFSET                # measured MAP (inHg)
        self.oil_press: float = _OIL_PRESS_OFFSET    # measured oil pressure (PSI)
        self.fuel_flow: float = _FUEL_FLOW_OFFSET    # measured fuel flow (L/h)
        self.iat: float = _IAT_OFFSET                # measured intake air temp (°C)
        self.vibration: float = _VIB_OFFSET          # measured vibration (g)

        # ── Fault & Manual Control State ──────────────────────────────
        self.fault_active: bool = False
        self.manual_override: bool = False
        self.manual_throttle: float = 0.6
        self.injected_fault: str = "NONE"

        # ── Independent Parameter Override State ──────────────────────
        self.override_flags: dict[str, bool] = {
            "rpm": False,
            "throttle": False,
            "egt": False,
            "cht": False,
            "oil_temp": False,
            "map": False,
            "oil_press": False,
            "fuel_flow": False,
            "iat": False,
            "vibration": False,
            "ambient_temp": False,
        }
        self.override_values: dict[str, float] = {
            "rpm": 3200.0,
            "throttle": 0.65,
            "egt": 650.0,
            "cht": 180.0,
            "oil_temp": 95.0,
            "map": 24.0,
            "oil_press": 50.0,
            "fuel_flow": 10.0,
            "iat": 30.0,
            "vibration": 0.45,
            "ambient_temp": 25.0,
        }

        # Every parameter advertised by the control API must accept an override.
        from layer1.adapter import PARAMETERS
        for name, spec in PARAMETERS.items():
            self.override_values.setdefault(name, spec.default)
            self.override_flags.setdefault(name, False)

    def set_atmospheric_profile(
        self,
        profile: str = "ISA_STANDARD",
        altitude: float = 0.0,
        ambient_temp: float = 25.0,
    ) -> None:
        """Set the active atmospheric profile and recalculate barometric pressure & density ratio."""
        aliases = {
            "ISA_STANDARD": ("ISA_STANDARD", 0.0, 25.0),
            "HIGH_ALTITUDE": ("HIGH_ALTITUDE", 20000.0, -15.0),
            "HIGH_ALTITUDE_THIN_AIR": ("HIGH_ALTITUDE", 20000.0, -15.0),
            "ENDURANCE": ("ENDURANCE_MISSION", 5000.0, 15.0),
            "ENDURANCE_MISSION": ("ENDURANCE_MISSION", 5000.0, 15.0),
            "HOT_WEATHER": ("HOT_WEATHER", 1000.0, 48.0),
            "HOT_WEATHER_OPERATION": ("HOT_WEATHER", 1000.0, 48.0),
            "HIGH_HEAT_DESERT": ("HOT_WEATHER", 1000.0, 48.0),
            "RAPID_TRANSITIONS": ("RAPID_TRANSITIONS", 3000.0, 20.0),
            "RAPID_THROTTLE_TRANSITIONS": ("RAPID_TRANSITIONS", 3000.0, 20.0),
            "ARCTIC_FREEZE": ("ARCTIC_FREEZE", 0.0, -25.0),
            "LIVE_STORM_TURBULENCE": ("LIVE_STORM_TURBULENCE", 3000.0, 20.0),
        }
        prof_upper = profile.upper().strip()
        if prof_upper not in aliases:
            raise ValueError(f"Unsupported atmospheric profile: {profile}")
        canonical, preset_altitude, preset_temp = aliases[prof_upper]
        self.atmospheric_profile = canonical
        # API-supplied altitude and temperature take precedence. Preserve the
        # historical profile presets when callers pass the untouched defaults.
        use_preset = (prof_upper != "ISA_STANDARD" and float(altitude) == 0.0
                      and float(ambient_temp) == 25.0)
        self.altitude = max(0.0, float(preset_altitude if use_preset else altitude))
        self.ambient_temp = float(preset_temp if use_preset else ambient_temp)

        # Recalculate ISA barometric pressure and air density ratio
        self.baro_pressure = _barometric_pressure_inhg(self.altitude)
        self.air_density_ratio = (self.baro_pressure / 29.92) * (298.15 / (273.15 + self.ambient_temp))

    # ------------------------------------------------------------------
    # Public Control API
    # ------------------------------------------------------------------

    def set_manual_control(self, throttle: float = 0.6, injected_fault: str = "NONE") -> None:
        """Enable manual simulator control and set live fault mode."""
        self.manual_override = True
        self.manual_throttle = float(np.clip(throttle, _THROTTLE_MIN, _THROTTLE_MAX))
        self.injected_fault = injected_fault.upper().strip()
        self.override_flags = {name: False for name in self.override_values}

    def set_multi_parameter_control(
        self,
        manual_override: bool = True,
        injected_fault: str = "NONE",
        override_flags: dict | None = None,
        override_values: dict | None = None,
    ) -> None:
        """Set independent parameter manual overrides."""
        self.manual_override = bool(manual_override)
        if injected_fault:
            self.injected_fault = injected_fault.upper().strip()
        # Treat each request as the complete active override selection. This
        # prevents a control from a previous dashboard mode remaining enabled.
        incoming_flags = override_flags if isinstance(override_flags, dict) else {}
        self.override_flags = {name: bool(incoming_flags.get(name, False))
                               for name in self.override_values}
        if override_values and isinstance(override_values, dict):
            for k, v in override_values.items():
                if k in self.override_values:
                    self.override_values[k] = float(v)

    def control_provenance(self) -> dict[str, dict[str, Any]]:
        """Describe simulator controls that replace individual sensor values."""
        provenance = {
            name: {"source": "manual_simulator_override", "measured": False,
                   "control_revision": self.control_revision}
            for name, enabled in self.override_flags.items() if enabled
        }
        if self.manual_override and not self.override_flags.get("throttle", False):
            provenance["throttle"] = {"source": "manual_simulator_control", "measured": False,
                                      "control_revision": self.control_revision}
        return provenance

    def clear_manual_control(self) -> None:
        """Clear manual control override and return to auto simulation."""
        self.manual_override = False
        self.injected_fault = "NONE"
        for k in self.override_flags:
            self.override_flags[k] = False

    def step(self) -> dict:
        """
        Advance the simulation by one time-step and return current telemetry.
        """
        # 1. Advance simulation clock
        self.t += self.dt

        # Environmental controls are model inputs. Apply them before pressure,
        # density, combustion, and sensor targets are calculated for this tick.
        if self.manual_override and self.override_flags.get("ambient_temp", False):
            self.ambient_temp = float(self.override_values.get("ambient_temp", self.ambient_temp))

        # 2. Determine throttle and RPM baseline
        mp_fault_mode = "NONE"
        if self.manual_override:
            if self.override_flags.get("throttle", False):
                self.manual_throttle = float(self.override_values.get("throttle", self.manual_throttle))
            self.throttle = float(np.clip(self.manual_throttle, _THROTTLE_MIN, _THROTTLE_MAX))
            if self.override_flags.get("rpm", False):
                rpm_true = float(self.override_values.get("rpm", _RPM_IDLE + _RPM_GAIN * self.throttle))
            else:
                rpm_true = _RPM_IDLE + _RPM_GAIN * self.throttle
        elif self._mission_profile_fn is not None:
            mp = self._mission_profile_fn(self.t)
            self.throttle = float(mp.get("throttle", _THROTTLE_CENTRE))
            rpm_true = float(mp.get("rpm_base", _RPM_IDLE + _RPM_GAIN * self.throttle))
            mp_fault_mode = mp.get("fault_mode", "NONE")
        else:
            self.throttle = self._compute_throttle(self.t)
            rpm_true = _RPM_IDLE + _RPM_GAIN * self.throttle

        # Determine effective active fault mode
        active_fault_mode = self.injected_fault if self.manual_override else mp_fault_mode

        # 3. Fault Injection Biases & Drag Determination
        egt_bias = 0.0
        cht_bias = 0.0
        oil_bias = 0.0
        map_bias = 0.0
        oil_press_bias = 0.0
        fuel_flow_bias = 0.0
        iat_bias = 0.0
        vib_bias = 0.0
        rpm_drag = 0.0

        if active_fault_mode == "OVERHEATING":
            egt_bias = 120.0
            cht_bias = 45.0
            iat_bias = 15.0
            self.fault_active = True
        elif active_fault_mode == "LUBRICATION_ISSUE":
            oil_bias = 35.0
            oil_press_bias = -28.0
            vib_bias = 1.2
            rpm_drag = 150.0   # mechanical friction drag
            self.fault_active = True
        elif active_fault_mode == "EXHAUST_LEAK":
            egt_bias = -60.0
            map_bias = -2.5
            self.fault_active = True
        elif active_fault_mode in ["FUEL_RESTRICTION", "FUEL_SYSTEM_RESTRICTION", "IMPROPER_INJECTION", "IMPROPER_INGESTION"]:
            fuel_flow_bias = -4.5
            egt_bias = 70.0    # lean combustion spike
            rpm_drag = 180.0
            vib_bias = 3.6     # severe vibration spike from improper ingestion & fuel mixture imbalance
            # Dynamic high-frequency vibration fluctuation during improper fuel ingestion
            t_curr = self.t
            vib_bias += float(0.85 * np.sin(42.0 * t_curr) + 0.65 * np.cos(87.0 * t_curr))
            self.fault_active = True
        elif active_fault_mode in ["MISFIRE", "CYLINDER_MISFIRE"]:
            vib_bias = 3.5     # severe unbalance vibration
            rpm_drag = 250.0
            egt_bias = -40.0
            self.fault_active = True
        elif self.enable_fault and not self.manual_override and self.t > _FAULT_TRIGGER_TIME:
            egt_bias = _EGT_FAULT_BIAS
            self.fault_active = True
        else:
            self.fault_active = False

        # 4. Effective RPM & measured RPM (after drag + noise)
        rpm_effective = max(0.0, rpm_true - rpm_drag)
        running = rpm_effective > 0.0
        self.rpm = max(0.0, rpm_effective + self._noise(_RPM_NOISE_STD)) if running else 0.0

        # 5. Calculate ISA Barometric Altitude Pressure & Air Density Ratio
        self.baro_pressure = _barometric_pressure_inhg(self.altitude)
        self.air_density_ratio = (self.baro_pressure / 29.92) * (298.15 / (273.15 + self.ambient_temp))
        sigma = self.air_density_ratio

        # 6. Environmental profile physics adjustments
        env_egt_bias = (1.0 - sigma) * 55.0  # thin air lean combustion EGT surge
        env_oil_press_bias = 0.0
        env_vib_bias = 0.0

        if self.atmospheric_profile == "ARCTIC_FREEZE":
            env_oil_press_bias = 22.0  # Cold oil viscosity pressure spike
        elif self.atmospheric_profile == "LIVE_STORM_TURBULENCE":
            env_vib_bias = float(0.20 + 0.15 * np.sin(0.4 * self.t) + 0.10 * np.sin(0.9 * self.t))

        # 7. True (noiseless) engine states calculated from effective engine speed & environmental profile
        d_amb = self.ambient_temp - 25.0
        # Combustion heat follows actual shaft speed. A commanded open throttle
        # cannot heat a stopped engine; stored heat dissipates over time.
        load = min(1.0, rpm_effective / _RPM_IDLE)
        egt_true = (self.ambient_temp + load * (
            _EGT_OFFSET - 25.0 + _EGT_THROTTLE_GAIN * self.throttle
            + _EGT_RPM_GAIN * (rpm_effective - _RPM_IDLE) + env_egt_bias))
        cht_true = self.ambient_temp + load * (
            _CHT_OFFSET - 25.0 + _CHT_EGT_COUPLING * (egt_true - _EGT_OFFSET))
        oil_temp_true = self.ambient_temp + load * (
            _OIL_TEMP_OFFSET - 25.0 + _OIL_TEMP_EGT_COUPLING * (egt_true - _EGT_OFFSET))
        
        map_max_alt = self.baro_pressure
        # MAP is an absolute pressure: throttle controls how close it gets to
        # ambient pressure, while altitude caps it at the local barometric
        # pressure. Air temperature affects charge density/fuel demand, not
        # the pressure reading itself at a fixed throttle and altitude.
        map_true = min(map_max_alt, _MAP_OFFSET + _MAP_THROTTLE_GAIN * self.throttle) if running else self.baro_pressure
        oil_press_true = max(0.0, _OIL_PRESS_OFFSET + _OIL_PRESS_RPM_GAIN * (rpm_effective - _RPM_IDLE) + _OIL_PRESS_TEMP_COUPLING * (self.oil_temp - _OIL_TEMP_OFFSET) + env_oil_press_bias) if running else 0.0
        fuel_flow_true = max(0.0, (_FUEL_FLOW_OFFSET + _FUEL_FLOW_THROTTLE_GAIN * self.throttle + _FUEL_FLOW_RPM_GAIN * (rpm_effective - _RPM_IDLE)) * max(0.4, sigma)) if running else 0.0
        iat_true = _IAT_OFFSET + _IAT_THROTTLE_GAIN * self.throttle + 0.95 * d_amb if running else self.ambient_temp
        vib_true = _VIB_OFFSET + _VIB_RPM_GAIN * (rpm_effective - _RPM_IDLE) + env_vib_bias if running else 0.0

        # 8. Add noise + fault bias to measured values
        def thermal(previous: float, target: float, tau: float, noise: float) -> float:
            alpha = 1.0 - math.exp(-self.dt / tau)
            updated = previous + alpha * (target - previous)
            return max(self.ambient_temp, updated + (self._noise(noise) * alpha if running else 0.0))

        self.egt = thermal(self.egt, egt_true + (egt_bias if running else 0.0), 8.0, _EGT_NOISE_STD)
        self.cht = thermal(self.cht, cht_true + (cht_bias if running else 0.0), 45.0, _CHT_NOISE_STD)
        self.oil_temp = thermal(self.oil_temp, oil_temp_true + (oil_bias if running else 0.0), 90.0, _OIL_TEMP_NOISE_STD)
        self.map = max(5.0, map_true + (map_bias + self._noise(_MAP_NOISE_STD) if running else 0.0))
        self.oil_press = max(0.0, oil_press_true + oil_press_bias + self._noise(_OIL_PRESS_NOISE_STD)) if running else 0.0
        self.fuel_flow = max(0.0, fuel_flow_true + fuel_flow_bias + self._noise(_FUEL_FLOW_NOISE_STD)) if running else 0.0
        self.iat = iat_true + (iat_bias + self._noise(_IAT_NOISE_STD) if running else 0.0)
        self.vibration = max(0.0, vib_true + vib_bias + self._noise(_VIB_NOISE_STD)) if running else 0.0

        # 9. Apply individual parameter manual overrides if enabled
        if self.manual_override:
            if self.override_flags.get("rpm", False):
                self.rpm = float(self.override_values.get("rpm", self.rpm))
            if self.override_flags.get("throttle", False):
                self.throttle = float(self.override_values.get("throttle", self.throttle))
            if self.override_flags.get("egt", False):
                self.egt = float(self.override_values.get("egt", self.egt))
            if self.override_flags.get("cht", False):
                self.cht = float(self.override_values.get("cht", self.cht))
            if self.override_flags.get("oil_temp", False):
                self.oil_temp = float(self.override_values.get("oil_temp", self.oil_temp))
            if self.override_flags.get("map", False):
                self.map = float(self.override_values.get("map", self.map))
            if self.override_flags.get("oil_press", False):
                self.oil_press = float(self.override_values.get("oil_press", self.oil_press))
            if self.override_flags.get("fuel_flow", False):
                self.fuel_flow = float(self.override_values.get("fuel_flow", self.fuel_flow))
            if self.override_flags.get("iat", False):
                self.iat = float(self.override_values.get("iat", self.iat))
            if self.override_flags.get("vibration", False):
                self.vibration = float(self.override_values.get("vibration", self.vibration))

        return self._build_telemetry()

    def reset(self) -> None:
        """Reset simulation time and fault state to initial conditions."""
        self.t = 0.0
        self.throttle = _THROTTLE_CENTRE
        self.rpm = _RPM_IDLE + _RPM_GAIN * _THROTTLE_CENTRE
        self.egt = _EGT_OFFSET
        self.cht = _CHT_OFFSET
        self.oil_temp = _OIL_TEMP_OFFSET
        self.map = _MAP_OFFSET
        self.oil_press = _OIL_PRESS_OFFSET
        self.fuel_flow = _FUEL_FLOW_OFFSET
        self.iat = _IAT_OFFSET
        self.vibration = _VIB_OFFSET
        self.fault_active = False
        self.clear_manual_control()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_throttle(self, t: float) -> float:
        """Return throttle position at time *t* (clamped to valid range)."""
        raw = (
            _THROTTLE_CENTRE
            + _THROTTLE_AMPLITUDE * np.sin(_THROTTLE_FREQ * t)
            + self._noise(_THROTTLE_NOISE_STD)
        )
        return float(np.clip(raw, _THROTTLE_MIN, _THROTTLE_MAX))

    def _noise(self, std: float) -> float:
        """Return a single Gaussian noise sample with zero mean and given std."""
        return float(self._rng.normal(loc=0.0, scale=std))

    def _build_telemetry(self) -> dict:
        """Package current state into the standard telemetry dictionary."""
        # Calculate Battery & Alternator Health
        running = self.rpm > 0.0
        v_bat = max(18.0, min(32.0, 28.0 + 0.0006 * (self.rpm - 2000.0) - (3.8 if self.injected_fault in ["LUBRICATION_ISSUE", "OVERHEATING"] else 0.0) + self._noise(0.12))) if running else 24.0
        a_alt = max(0.0, min(80.0, 45.0 + 0.004 * (self.rpm - 2000.0) - (14.0 if self.injected_fault in ["LUBRICATION_ISSUE"] else 0.0) + self._noise(0.5))) if running else 0.0
        e_health = max(0.0, min(100.0, 100.0 - max(0.0, 27.5 - v_bat) * 18.0))

        # Calculate Fuel Injection Timing & Pulse Duration
        inj_time = 22.5 + 0.0015 * (self.rpm - 2000.0) + 4.5 * self.throttle + (-4.0 if self.injected_fault in ["CYLINDER_MISFIRE", "FUEL_RESTRICTION"] else 0.0) + self._noise(0.15) if running else 0.0
        inj_dur = max(1.0, 4.20 + 3.6 * self.throttle + (-1.2 if self.injected_fault in ["FUEL_RESTRICTION"] else 0.0) + self._noise(0.05)) if running else 0.0
        fuel_p = max(5.0, 42.0 + 3.0 * self.throttle - (22.0 if self.injected_fault in ["FUEL_RESTRICTION"] else 0.0) + self._noise(0.4)) if running else 0.0

        telemetry = {
            "timestamp": round(self.t, 4),
            "throttle": round(self.throttle, 4),
            "rpm": round(self.rpm, 2),
            "egt": round(self.egt, 2),       # Exhaust Gas Temperature (°C)
            "cht": round(self.cht, 2),       # Cylinder Head Temperature (°C)
            "oil_temp": round(self.oil_temp, 2),
            "map": round(self.map, 2),       # Manifold Absolute Pressure (inHg)
            "oil_press": round(self.oil_press, 2), # Oil Pressure (PSI)
            "fuel_flow": round(self.fuel_flow, 2), # Fuel Flow Rate (L/h)
            "iat": round(self.iat, 2),       # Intake Air Temperature (°C)
            "vibration": round(self.vibration, 3), # Vibration Level (g RMS)
            "battery_voltage": round(v_bat, 2),
            "alternator_current": round(a_alt, 1),
            "electrical_health": round(e_health, 1),
            "injection_timing": round(inj_time, 2),
            "injection_duration": round(inj_dur, 2),
            "fuel_pressure": round(fuel_p, 1),
            "fault_active": self.fault_active,
            "ambient_temp": round(self.ambient_temp, 1),
            "atmospheric_profile": self.atmospheric_profile,
            "altitude": round(self.altitude, 1),
            "baro_pressure": round(self.baro_pressure, 2),
            "air_density_ratio": round(self.air_density_ratio, 4),
            "control_revision": self.control_revision,
            "override_values": {name: self.override_values[name] for name, enabled in self.override_flags.items() if enabled},
            "provenance": self.control_provenance(),
        }
        if self.manual_override:
            for name, enabled in self.override_flags.items():
                if enabled and name in telemetry:
                    telemetry[name] = self.override_values[name]
        return telemetry



# ---------------------------------------------------------------------------
# Quick self-test  (run: python -m app.simulator)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    sim = EngineSimulator(dt=0.5, seed=42)
    print(f"{'t':>8} {'throttle':>9} {'rpm':>8} {'egt':>8} {'cht':>8} {'oil':>8} {'fault':>6}")
    print("-" * 62)

    for _ in range(160):   # 80 seconds of simulated time
        telem = sim.step()
        if telem["timestamp"] % 5.0 < sim.dt:   # print every ~5 s
            print(
                f"{telem['timestamp']:>8.1f} "
                f"{telem['throttle']:>9.3f} "
                f"{telem['rpm']:>8.1f} "
                f"{telem['egt']:>8.1f} "
                f"{telem['cht']:>8.1f} "
                f"{telem['oil_temp']:>8.1f} "
                f"{'YES' if telem['fault_active'] else 'no':>6}"
            )
