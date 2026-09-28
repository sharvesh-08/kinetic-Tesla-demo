"""React API contracts, timestamp precision and advertised overrides."""
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app import main
from app.physics_model import predict_expected
from app.simulator import EngineSimulator
from layer1.adapter import PARAMETERS, to_layer1_events

class RedisSnapshot:
    def __init__(self, stale=False):
        self.stamp = time.time_ns() - (20_000_000_000 if stale else 0)
    async def xrevrange(self, stream, count=1):
        item = {"session_id":"test-session", "control_revision":3,
                "timestamp_ns":self.stamp, "window_end_ns":self.stamp,
                "classification_available":False,"classification_unavailable_reason":"Unverified artifact",
                "features":{"a":1,"b":2},"window_length_s":5}
        return [("1-0", {"payload":json.dumps(item)})]

class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def snapshot(self, stale=False):
        sim=EngineSimulator(seed=7)
        with patch.multiple(main, _simulator=sim, _twin=SimpleNamespace(session_id="test-session",redis=RedisSnapshot(stale)),
                            _latest_state={"timestamp":1.5,"session_id":"test-session"},
                            _latest_state_received_at=time.monotonic(),_control_revision=3):
            return await main.dashboard_state()
    async def test_snapshot_exposes_real_layer6_gate_and_feature_count(self):
        result=await self.snapshot()
        self.assertEqual(result['window']['feature_count'],2)
        self.assertFalse(result['layer6']['classification_available'])
        self.assertEqual(result['layer6']['classification_unavailable_reason'],'Unverified artifact')
        self.assertIsInstance(result['layer6']['timestamp_ns'],str)
        self.assertEqual(result['control_revision'],3)
    async def test_stale_layer6_and_windows_are_not_presented_as_current(self):
        result=await self.snapshot(True)
        self.assertEqual(result['layer6'],{})
        self.assertIsNone(result['window']['feature_count'])
    def test_every_advertised_sensor_override_reaches_simulator_telemetry(self):
        sim=EngineSimulator(seed=7)
        values={name:spec.default for name,spec in PARAMETERS.items()}
        values.update(rpm=3100,battery_voltage=24.5,injection_duration=3.5)
        sim.set_multi_parameter_control(True,'NONE',{key:True for key in values},values)
        telemetry=sim.step()
        for key,value in values.items():
            self.assertEqual(telemetry[key],value,key)
            self.assertEqual(telemetry['provenance'][key]['source'],'manual_simulator_override')
    def test_zero_rpm_cools_engine_and_stops_shaft_driven_sensors(self):
        sim = EngineSimulator(dt=0.5, seed=7)
        for _ in range(120):
            sim.step()
        before = sim.step()
        sim.set_multi_parameter_control(override_flags={'rpm': True}, override_values={'rpm': 0})
        first = sim.step()
        later = first
        for _ in range(120):
            later = sim.step()
        for channel in ('egt', 'cht', 'oil_temp'):
            self.assertGreater(before[channel], first[channel], channel)
            self.assertGreater(first[channel], later[channel], channel)
            self.assertGreaterEqual(later[channel], sim.ambient_temp, channel)
        for channel in ('rpm', 'oil_press', 'fuel_flow', 'vibration',
                        'alternator_current', 'injection_duration', 'injection_timing'):
            self.assertEqual(first[channel], 0, channel)
        self.assertAlmostEqual(first['map'], first['baro_pressure'])
    def test_rpm_drives_engine_thermal_and_shaft_sensors(self):
        def at_rpm(rpm):
            sim = EngineSimulator(dt=0.5, seed=11)
            sim.set_multi_parameter_control(override_flags={'rpm': True}, override_values={'rpm': rpm})
            for _ in range(360):
                telemetry = sim.step()
            return telemetry

        low, high = at_rpm(2500), at_rpm(4500)
        for channel in ('egt', 'cht', 'oil_temp', 'oil_press', 'fuel_flow',
                        'vibration', 'alternator_current', 'battery_voltage'):
            self.assertGreater(high[channel], low[channel], channel)
        # At fixed throttle MAP is capped by manifold/ambient pressure; it is
        # not expected to rise monotonically with RPM like oil pressure does.
        self.assertLessEqual(high['map'], high['baro_pressure'])

    def test_ambient_and_altitude_change_coupled_sensor_targets(self):
        cool = EngineSimulator(dt=0.5, seed=13)
        hot = EngineSimulator(dt=0.5, seed=13)
        hot.set_multi_parameter_control(override_flags={'ambient_temp': True},
                                        override_values={'ambient_temp': 45})
        for _ in range(360):
            cool_t, hot_t = cool.step(), hot.step()
        self.assertGreater(hot_t['iat'], cool_t['iat'])
        self.assertGreater(hot_t['egt'], cool_t['egt'])
        self.assertGreater(hot_t['cht'], cool_t['cht'])
        self.assertGreater(hot_t['oil_temp'], cool_t['oil_temp'])

        sea_level = EngineSimulator(seed=17)
        high_altitude = EngineSimulator(seed=17)
        sea_level.set_manual_control(throttle=0.9)
        high_altitude.set_manual_control(throttle=0.9)
        high_altitude.set_atmospheric_profile('ISA_STANDARD', altitude=10000, ambient_temp=25)
        sea_t, altitude_t = sea_level.step(), high_altitude.step()
        self.assertLess(altitude_t['baro_pressure'], sea_t['baro_pressure'])
        self.assertLess(altitude_t['map'], sea_t['map'])
        self.assertLess(altitude_t['air_density_ratio'], sea_t['air_density_ratio'])
        self.assertAlmostEqual(altitude_t['baro_pressure'], 20.58, delta=0.15)

    def test_physics_prediction_matches_engine_shutdown_and_environment(self):
        stopped = predict_expected(rpm=0, throttle=0.7, ambient_temp=30, altitude=0)
        self.assertEqual(stopped.egt_expected, 30)
        self.assertEqual(stopped.cht_expected, 30)
        self.assertEqual(stopped.oil_temp_expected, 30)
        self.assertEqual(stopped.oil_press_expected, 0)
        self.assertEqual(stopped.fuel_flow_expected, 0)
        self.assertEqual(stopped.vibration_expected, 0)

        standard = predict_expected(rpm=4000, throttle=0.9, ambient_temp=25, altitude=0)
        hot = predict_expected(rpm=4000, throttle=0.9, ambient_temp=45, altitude=0)
        high_altitude = predict_expected(rpm=4000, throttle=0.9, ambient_temp=25, altitude=3048)
        self.assertGreater(hot.iat_expected, standard.iat_expected)
        self.assertGreater(hot.cht_expected, standard.cht_expected)
        self.assertLess(high_altitude.baro_pressure, standard.baro_pressure)
        self.assertLess(high_altitude.map_expected, standard.map_expected)

    def test_ambient_temperature_is_carried_into_operating_context(self):
        events = to_layer1_events({"timestamp": 10.0, "rpm": 2500,
                                  "ambient_temp": 42.0, "altitude": 10000.0},
                                 session_id="environment-test")
        context = events[0]["context"]["operating_context"]
        self.assertEqual(context["ambient_temp_degC"], 42.0)
        self.assertAlmostEqual(context["altitude_m"], 3048.0)
    def test_nanoseconds_round_trip_without_javascript_precision_loss(self):
        value=1790337517975123456
        self.assertEqual(main._dashboard_timestamps({'frame':{'master_timestamp_ns':value}})['frame']['master_timestamp_ns'],str(value))
