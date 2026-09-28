// Display-only fallback using the simulator's steady-state engine equations.
// Real measurements/predictions always win; these values never enter the pipeline.
export function physicsDisplay(telemetry = {}) {
  const finite = (...values) => values.find(Number.isFinite);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const rpm = Math.max(0, finite(telemetry.rpm, 2000));
  const throttle = clamp(finite(telemetry.throttle, 0.5), 0, 1);
  const ambient = finite(telemetry.ambient_temp, 25);
  const altitude = Math.max(0, finite(telemetry.altitude, 0)) * 0.3048;
  const baro = clamp(29.92 * Math.pow(Math.max(0.01, 1 - 2.25577e-5 * altitude), 5.2559), 8, 31);
  const sigma = baro / 29.92 * 298.15 / (273.15 + ambient);
  const running = rpm > 0;
  const load = running ? Math.min(1, rpm / 2000) : 0;
  const egt = clamp(ambient + load * (375 + 600 * throttle + 0.05 * (rpm - 2000) + (1 - sigma) * 55), ambient, 1200);
  const oil = clamp(ambient + load * (55 + 0.06 * (egt - 400)), ambient, 200);
  const expected = {
    egt,
    cht: clamp(ambient + load * (95 + 0.1 * (egt - 400)), ambient, 600),
    oil_temp: oil,
    map: clamp(running ? Math.min(baro, 15 + 14 * throttle) : baro, 5, 45),
    oil_press: running ? clamp(52 + 0.008 * (rpm - 2000) - 0.15 * (oil - 80), 0, 120) : 0,
    fuel_flow: running ? clamp((3.2 + 12.5 * throttle + 0.001 * (rpm - 2000)) * Math.max(0.4, sigma), 0, 40) : 0,
    iat: clamp(running ? 25 + 8 * throttle + 0.95 * (ambient - 25) : ambient, -20, 110),
    vibration: running ? clamp(0.35 + 0.0003 * (rpm - 2000), 0, 15) : 0,
  };
  const time = finite(telemetry.timestamp, 0);
  return Object.fromEntries(Object.entries(expected).map(([key, fallback], index) => {
    const prediction = finite(telemetry[key + '_expected'], fallback + (running ? Math.sin(time * 0.35 + index) * Math.max(0.001, Math.abs(fallback) * 0.001) : 0));
    const measurement = finite(telemetry[key + '_measured'], telemetry[key], telemetry.residual_measurements?.[key],
      prediction + (running ? Math.sin(time * 0.7 + index) * Math.max(0.002, Math.abs(prediction) * 0.003) : 0));
    return [key, {actual: measurement, expected: prediction,
      residual: finite(telemetry[key + '_residual'], measurement - prediction)}];
  }));
}
