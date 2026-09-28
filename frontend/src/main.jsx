import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { request, appendSample, format, currentAssessment, hasSessionCache, displayedRul } from './api';
import './style.css';
import { physicsDisplay } from './physicsDisplay';

const names = {
  rpm: 'Engine Speed',
  throttle: 'Throttle Command',
  egt: 'Exhaust Gas Temp',
  cht: 'Cylinder Head Temp',
  oil_temp: 'Oil Temperature',
  map: 'Manifold Pressure',
  oil_press: 'Oil Pressure',
  fuel_flow: 'Fuel Flow Rate',
  iat: 'Intake Air Temp',
  vibration: 'Engine Vibration',
  ambient_temp: 'Ambient Temp',
  battery_voltage: 'Bus Voltage',
  alternator_current: 'Alternator Current',
  electrical_health: 'Electrical Health',
  injection_timing: 'Injection Timing',
  injection_duration: 'Injection Duration',
  fuel_pressure: 'Fuel Pressure'
};

const faults = [
  'NONE',
  'OVERHEATING',
  'LUBRICATION_ISSUE',
  'EXHAUST_LEAK',
  'FUEL_RESTRICTION',
  'CYLINDER_MISFIRE',
  'IMPROPER_INGESTION'
];

const profiles = [
  'ISA_STANDARD',
  'HIGH_ALTITUDE',
  'ENDURANCE_MISSION',
  'HOT_WEATHER',
  'ARCTIC_FREEZE',
  'LIVE_STORM_TURBULENCE'
];

const SENSOR_CHANNEL_LABELS = {
  rpm: 'RPM',
  map_kPa: 'MPA',
  map: 'MPA',
  Tm_K_k: 'TMEP',
  cht_degC: 'CHT',
  cht: 'CHT',
  egt_degC: 'EGT',
  egt: 'EGT',
  oil_press_kPa: 'OIL PRESS',
  oil_press: 'OIL PRESS',
  oil_temp_degC: 'OIL TEMP',
  oil_temp: 'OIL TEMP',
  fuel_flow_Lph: 'FUEL FLOW',
  fuel_flow: 'FUEL FLOW',
  ambient_temp_degC: 'AMBIENT TEMP',
  ambient_temp: 'AMBIENT TEMP',
  ambient_press_kPa: 'AMBIENT PRESS',
  throttle_pct: 'THROTTLE',
  batt_volts: 'BATT VOLTS',
  alt_amps: 'ALT AMPS',
  vibration_rms_g: 'VIBRATION'
};

const label = s => String(s ?? 'Unavailable').replaceAll('_', ' ');

function Badge({ children, tone = '' }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function Metric({ name, value, unit = '', note }) {
  return (
    <div className="metric">
      <small>{name}</small>
      <strong>
        {value}
        {unit && typeof value === 'string' && /^-?\d+(?:\.\d+)?$/.test(value) && <span>{unit}</span>}
      </strong>
      {note && <small>{note}</small>}
    </div>
  );
}

const TELEMETRY_PARAMETERS = [
  { key: 'rpm', title: 'RPM', unit: 'RPM', category: 'engine', minLimit: 0, maxLimit: 5800 },
  { key: 'map', title: 'Manifold Absolute Pressure', unit: 'inHg', category: 'engine', minLimit: 10, maxLimit: 35 },
  { key: 'fuel_flow', title: 'Fuel Flow Rate', unit: 'L/h', category: 'engine', minLimit: 5, maxLimit: 45 },
  { key: 'egt', title: 'Exhaust Gas Temperature', unit: '°C', category: 'thermal', minLimit: 400, maxLimit: 850 },
  { key: 'vibration', title: 'Engine Vibration', unit: 'g RMS', category: 'engine', minLimit: 0, maxLimit: 2.5 },
  { key: 'cht', title: 'Cylinder Head Temperature', unit: '°C', category: 'thermal', minLimit: 80, maxLimit: 220 },
  { key: 'oil_temp', title: 'Oil Temperature', unit: '°C', category: 'thermal', minLimit: 60, maxLimit: 130 },
  { key: 'oil_press', title: 'Oil Pressure', unit: 'PSI', category: 'thermal', minLimit: 30, maxLimit: 75 },
  { key: 'iat', title: 'Intake Air Temperature', unit: '°C', category: 'thermal', minLimit: -10, maxLimit: 60 }
];

function Trend({ history, field, title, unit, category, minLimit, maxLimit }) {
  const [hover, setHover] = useState(null);

  const valOf = x => {
    if (!x) return undefined;
    if (field.endsWith('_residual')) {
      const base = field.replace('_residual', '');
      if (x.residual_measurements && Number.isFinite(x.residual_measurements[field])) return x.residual_measurements[field];
      if (x.residual_measurements && Number.isFinite(x.residual_measurements[base + '_residual'])) return x.residual_measurements[base + '_residual'];
      if (x.residual_measurements && Number.isFinite(x.residual_measurements[base])) return x.residual_measurements[base];
      if (Number.isFinite(x[field])) return x[field];
      if (Number.isFinite(x[base + '_residual'])) return x[base + '_residual'];
    }
    if (Number.isFinite(x[field + '_measured'])) return x[field + '_measured'];
    if (Number.isFinite(x[field])) return x[field];
    if (x.residual_measurements && Number.isFinite(x.residual_measurements[field])) return x.residual_measurements[field];
    if (field === 'battery_voltage' && Number.isFinite(x.batt_volts)) return x.batt_volts;
    if (field === 'alternator_current' && Number.isFinite(x.alt_amps)) return x.alt_amps;
    return undefined;
  };

  const samples = history.map((x, i) => ({ i, val: valOf(x), ts: x.timestamp })).filter(p => Number.isFinite(p.val));
  const ys = samples.map(p => p.val);
  const lo = ys.length ? Math.min(...ys) : 0;
  const hi = ys.length ? Math.max(...ys) : 100;
  const mid = (hi + lo) / 2;
  const span = hi - lo || 1;
  const avg = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;

  const tStart = samples[0]?.ts ?? 0;
  const tEnd = samples.at(-1)?.ts ?? 0;
  const tMid = (tStart + tEnd) / 2;

  const latestVal = samples.at(-1)?.val;
  const prevVal = samples.at(-2)?.val;
  const delta = (latestVal !== undefined && prevVal !== undefined) ? latestVal - prevVal : 0;

  let status = 'NOMINAL';
  let statusTone = 'good';
  if (latestVal !== undefined && ((maxLimit != null && latestVal > maxLimit) || (minLimit != null && latestVal < minLimit))) {
    status = 'OUT OF BOUNDS';
    statusTone = 'bad';
  } else if (field.endsWith('_residual') && Math.abs(latestVal ?? 0) > (maxLimit ? maxLimit * 0.7 : 5)) {
    status = 'HIGH RESIDUAL';
    statusTone = 'warning';
  }

  const paddingLeft = 60;
  const paddingRight = 20;
  const paddingTop = 22;
  const paddingBottom = 32;
  const width = 600;
  const height = 165;
  const graphW = width - paddingLeft - paddingRight;
  const graphH = height - paddingTop - paddingBottom;

  const numPoints = Math.max(1, history.length - 1);
  const getX = i => paddingLeft + (i / numPoints) * graphW;
  const getY = val => (height - paddingBottom) - ((val - lo) / span) * graphH;

  const path = samples.map((p, n) => {
    const x = getX(p.i);
    const y = getY(p.val);
    return `${n ? 'L' : 'M'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');

  const areaPath = samples.length
    ? `${path} L ${getX(samples.at(-1).i).toFixed(1)} ${height - paddingBottom} L ${getX(samples[0].i).toFixed(1)} ${height - paddingBottom} Z`
    : '';

  const lastPt = samples.at(-1);
  const lastX = lastPt ? getX(lastPt.i) : width - paddingRight;
  const lastY = lastPt ? getY(lastPt.val) : height / 2;

  const strokeColor = field.endsWith('_residual')
    ? '#6366f1'
    : statusTone === 'bad'
    ? '#ef4444'
    : statusTone === 'warning'
    ? '#f59e0b'
    : '#0d9488';

  const handleMouseMove = e => {
    if (!samples.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const svgX = (mouseX / rect.width) * width;
    let closest = samples[0];
    let minDiff = Math.abs(getX(closest.i) - svgX);
    for (let s of samples) {
      const diff = Math.abs(getX(s.i) - svgX);
      if (diff < minDiff) {
        minDiff = diff;
        closest = s;
      }
    }
    setHover({
      x: getX(closest.i),
      y: getY(closest.val),
      val: closest.val,
      ts: closest.ts
    });
  };

  const handleMouseLeave = () => setHover(null);

  return (
    <div className="trend-card">
      <div className="trend-head">
        <div className="trend-title-block">
          <span className="trend-label">
            {title} <small className="mono">[{field}]</small>
          </span>
          <div className="trend-curr-wrapper">
            <span className="trend-curr" style={{ color: strokeColor }}>
              {format(latestVal)} <small>{unit}</small>
            </span>
            {delta !== 0 && (
              <span className={`trend-delta ${delta > 0 ? 'up' : 'down'}`}>
                {delta > 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(1)}
              </span>
            )}
          </div>
        </div>
        <div className="trend-head-right">
          <Badge tone={statusTone}>{status}</Badge>
        </div>
      </div>

      <div className="trend-readings-strip">
        <span><b>MIN:</b> {format(lo)} {unit}</span>
        <span><b>MAX:</b> {format(hi)} {unit}</span>
        <span><b>AVG:</b> {format(avg)} {unit}</span>
        <span><b>SPAN:</b> {format(span)} {unit}</span>
      </div>

      <div className="trend-chart-box">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`${title} telemetry history plot`}
          preserveAspectRatio="none"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <defs>
            <linearGradient id={`grad-${field}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={strokeColor} stopOpacity="0.25" />
              <stop offset="100%" stopColor={strokeColor} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid Lines & Y-Axis levels */}
          <line x1={paddingLeft} y1={paddingTop} x2={width - paddingRight} y2={paddingTop} stroke="#e2e8f0" strokeWidth="1" />
          <line x1={paddingLeft} y1={paddingTop + graphH * 0.25} x2={width - paddingRight} y2={paddingTop + graphH * 0.25} stroke="#f1f5f9" strokeDasharray="3 3" strokeWidth="1" />
          <line x1={paddingLeft} y1={paddingTop + graphH * 0.5} x2={width - paddingRight} y2={paddingTop + graphH * 0.5} stroke="#cbd5e1" strokeDasharray="4 3" strokeWidth="1" />
          <line x1={paddingLeft} y1={paddingTop + graphH * 0.75} x2={width - paddingRight} y2={paddingTop + graphH * 0.75} stroke="#f1f5f9" strokeDasharray="3 3" strokeWidth="1" />
          <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} stroke="#e2e8f0" strokeWidth="1" />

          {/* Grid Lines & X-Axis time ticks */}
          <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} stroke="#cbd5e1" strokeWidth="1.5" />
          <line x1={paddingLeft + graphW * 0.25} y1={paddingTop} x2={paddingLeft + graphW * 0.25} y2={height - paddingBottom} stroke="#f1f5f9" strokeDasharray="3 3" strokeWidth="1" />
          <line x1={paddingLeft + graphW * 0.5} y1={paddingTop} x2={paddingLeft + graphW * 0.5} y2={height - paddingBottom} stroke="#e2e8f0" strokeDasharray="4 3" strokeWidth="1" />
          <line x1={paddingLeft + graphW * 0.75} y1={paddingTop} x2={paddingLeft + graphW * 0.75} y2={height - paddingBottom} stroke="#f1f5f9" strokeDasharray="3 3" strokeWidth="1" />
          <line x1={width - paddingRight} y1={paddingTop} x2={width - paddingRight} y2={height - paddingBottom} stroke="#cbd5e1" strokeWidth="1.5" />

          {/* Y-Axis Value Labels */}
          <text x={paddingLeft - 7} y={paddingTop + 4} textAnchor="end" fill="#475569" fontSize="9.5" fontWeight="700" fontFamily="ui-monospace, monospace">{format(hi)}</text>
          <text x={paddingLeft - 7} y={paddingTop + graphH * 0.5 + 3} textAnchor="end" fill="#64748b" fontSize="8.5" fontWeight="600" fontFamily="ui-monospace, monospace">{format(mid)}</text>
          <text x={paddingLeft - 7} y={height - paddingBottom + 3} textAnchor="end" fill="#475569" fontSize="9.5" fontWeight="700" fontFamily="ui-monospace, monospace">{format(lo)}</text>

          {/* Y-Axis Title Label */}
          <text x={14} y={paddingTop + graphH / 2} textAnchor="middle" fill="#64748b" fontSize="8.5" fontWeight="700" fontFamily="Inter, sans-serif" transform={`rotate(-90 14 ${paddingTop + graphH / 2})`}>
            {unit ? `${unit}` : 'Value'}
          </text>

          {/* Graph Wave & Area Fill */}
          {areaPath && <path d={areaPath} fill={`url(#grad-${field})`} />}
          {path && <path d={path} fill="none" stroke={strokeColor} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />}

          {/* Current Live Marker Dot */}
          {samples.length > 0 && (
            <>
              <circle cx={lastX} cy={lastY} r="4.5" fill={strokeColor} stroke="#ffffff" strokeWidth="1.5" />
              <circle cx={lastX} cy={lastY} r="7.5" fill="none" stroke={strokeColor} strokeWidth="1" opacity="0.6" />
            </>
          )}

          {/* X-Axis Time Ticks & Labels */}
          <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} stroke="#64748b" strokeWidth="1.5" />
          <line x1={paddingLeft} y1={height - paddingBottom} x2={paddingLeft} y2={height - paddingBottom + 5} stroke="#64748b" strokeWidth="1.5" />
          <line x1={paddingLeft + graphW * 0.5} y1={height - paddingBottom} x2={paddingLeft + graphW * 0.5} y2={height - paddingBottom + 5} stroke="#94a3b8" strokeWidth="1" />
          <line x1={width - paddingRight} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom + 5} stroke={strokeColor} strokeWidth="1.5" />

          <text x={paddingLeft} y={height - 10} textAnchor="start" fill="#64748b" fontSize="9" fontWeight="600" fontFamily="ui-monospace, monospace">T+{format(tStart, 1)}s</text>
          <text x={paddingLeft + graphW * 0.5} y={height - 10} textAnchor="middle" fill="#94a3b8" fontSize="8.5" fontWeight="500" fontFamily="ui-monospace, monospace">T+{format(tMid, 1)}s</text>
          <text x={width - paddingRight} y={height - 10} textAnchor="end" fill={strokeColor} fontSize="9.5" fontWeight="700" fontFamily="ui-monospace, monospace">T+{format(tEnd, 1)}s (LIVE)</text>

          {/* Interactive Hover Crosshair & Tooltip */}
          {hover && (
            <>
              <line x1={hover.x} y1={paddingTop} x2={hover.x} y2={height - paddingBottom} stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="3 2" />
              <circle cx={hover.x} cy={hover.y} r="5" fill="#3b82f6" stroke="#ffffff" strokeWidth="2" />

              <g transform={`translate(${Math.min(width - 120, Math.max(paddingLeft, hover.x - 50))}, ${Math.max(paddingTop, hover.y - 34)})`}>
                <rect width="105" height="28" rx="5" fill="#0f172a" opacity="0.92" />
                <text x="52" y="13" textAnchor="middle" fill="#ffffff" fontSize="9.5" fontWeight="700" fontFamily="ui-monospace, monospace">
                  {format(hover.val)} {unit}
                </text>
                <text x="52" y="23" textAnchor="middle" fill="#94a3b8" fontSize="8" fontWeight="500" fontFamily="ui-monospace, monospace">
                  Time: T+{format(hover.ts, 1)}s
                </text>
              </g>
            </>
          )}
        </svg>
      </div>

      <div className="trend-footer">
        <span><b>Y-AXIS:</b> {title} ({unit || 'raw'})</span>
        <span><b>X-AXIS:</b> Time (s) · {samples.length} pts</span>
      </div>
    </div>
  );
}

function TrendsPanel({ history }) {
  const [category, setCategory] = useState('all');
  const [search, setSearch] = useState('');
  const [cols, setCols] = useState(2);

  const categories = [
    { id: 'all', label: 'All Parameters' },
    { id: 'engine', label: 'Engine Dynamics' },
    { id: 'thermal', label: 'Thermal Systems' }
  ];

  const filteredParams = TELEMETRY_PARAMETERS.filter(p => {
    const matchesCat = category === 'all' || p.category === category;
    const matchesSearch = !search || p.title.toLowerCase().includes(search.toLowerCase()) || p.key.toLowerCase().includes(search.toLowerCase());
    return matchesCat && matchesSearch;
  });

  return (
    <div className="panel" data-tab="04 05 all">
      <div className="panel-title">
        <div>
          <h2>Live Telemetry & Parameter Trend Oscilloscope</h2>
        </div>
      </div>

      <div className="oscilloscope-controls">
        <div className="category-buttons">
          {categories.map(c => (
            <button
              key={c.id}
              type="button"
              className={`category-btn ${category === c.id ? 'active' : ''}`}
              onClick={() => setCategory(c.id)}
            >
              {c.label} {c.id !== 'all' ? `(${TELEMETRY_PARAMETERS.filter(p => p.category === c.id).length})` : `(${TELEMETRY_PARAMETERS.length})`}
            </button>
          ))}
        </div>
        <div style={{ maxWidth: '240px', width: '100%' }}>
          <input
            type="text"
            placeholder="Search parameters..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className={`oscilloscope-grid grid-cols-${cols}`}>
        {filteredParams.map(p => (
          <Trend
            key={p.key}
            history={history}
            field={p.key}
            title={p.title}
            unit={p.unit}
            category={p.category}
            minLimit={p.minLimit}
            maxLimit={p.maxLimit}
          />
        ))}
        {filteredParams.length === 0 && (
          <div className="empty" style={{ gridColumn: '1 / -1' }}>
            No telemetry parameters match the current category or search filter.
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [packet, setPacket] = useState(null);
  const [diagnosticsPacket, setDiagnosticsPacket] = useState(null);
  const latestPacket = useRef(null);
  const [history, setHistory] = useState([]);
  const [catalog, setCatalog] = useState({});
  const [online, setOnline] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [revision, setRevision] = useState(null);
  const [lastRiskAssessment, setLastRiskAssessment] = useState(null);
  const [lastLayer6Assessment, setLastLayer6Assessment] = useState(null);
  const [maintenanceAdvice, setMaintenanceAdvice] = useState(null);
  const [advisoryBusy, setAdvisoryBusy] = useState(false);
  const [advisoryError, setAdvisoryError] = useState('');

  const [manual, setManual] = useState(false);
  const [fault, setFault] = useState('NONE');
  const [flags, setFlags] = useState({});
  const [values, setValues] = useState({});
  const [environment, setEnvironment] = useState({ profile: 'ISA_STANDARD', altitude: 0, ambient_temp: 25 });
  const [missions, setMissions] = useState([]);
  const [mission, setMission] = useState('');
  const [search, setSearch] = useState('');

  const initialized = useRef(false);
  const viewer = useRef(null);
  const [viewerReady, setViewerReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer;
    const controller = new AbortController();

    async function poll() {
      try {
        const data = await request('/state', undefined, AbortSignal.any([controller.signal, AbortSignal.timeout(3000)]));
        if (cancelled) return;
        latestPacket.current = data;
        setPacket(data);
        setDiagnosticsPacket(previous => previous || data);
        setOnline(true);
        setError('');
        setHistory(h => appendSample(h, data.telemetry));

        if (data.telemetry?.session_id && initialized.current !== data.telemetry.session_id) {
          initialized.current = data.telemetry.session_id;
          setRevision(null);
          setLastRiskAssessment(null);
          setLastLayer6Assessment(null);
          setAdvisoryError('');
          setCatalog(data.parameters || {});
          setManual(data.controls.manual_override);
          setFault(data.controls.injected_fault || 'NONE');
          setFlags(data.controls.override_flags || {});
          setValues(data.controls.override_values || {});
          setEnvironment({
            profile: data.environment.atmospheric_profile,
            altitude: data.environment.altitude,
            ambient_temp: data.environment.ambient_temp
          });
          setMissions(data.missions || []);
          setMission(data.missions?.[0]?.name || data.missions?.[0] || '');
        }
      } catch (e) {
        if (cancelled) return;
        setOnline(false);
        setError(e.message);
      } finally {
        if (!cancelled) timer = setTimeout(poll, 500);
      }
    }
    poll();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      if (latestPacket.current) setDiagnosticsPacket(latestPacket.current);
    }, 10000);
    return () => clearInterval(timer);
  }, []);

  const telemetry = packet?.telemetry || {};
  const diagnosticTelemetry = diagnosticsPacket?.telemetry || {};
  const assessment = diagnosticTelemetry.pipeline || {};
  const layer6 = diagnosticsPacket?.layer6 || {};
  const physicsRows = physicsDisplay(telemetry);
  const frame = telemetry.layer2_sensor_frame || {};
  const fresh = online && packet?.simulation?.running && Number.isFinite(packet?.telemetry_age_ms) && packet.telemetry_age_ms < 4000;
  const layer6Current = layer6.session_id === diagnosticTelemetry.session_id &&
    layer6.control_revision === diagnosticsPacket?.control_revision && Object.keys(layer6.black_box || {}).length > 0;
  const matched = currentAssessment(diagnosticsPacket, diagnosticsPacket?.control_revision);
  const pending = !fresh || busy || !currentAssessment(packet, revision);
  useEffect(() => {
    if (layer6Current) {
      setLastLayer6Assessment({ sessionId: diagnosticTelemetry.session_id, assessment: layer6 });
    }
  }, [layer6Current, layer6, diagnosticTelemetry.session_id]);
  useEffect(() => {
    if (matched && diagnosticTelemetry.session_id &&
        (Number.isFinite(assessment.risk_fusion?.engine_risk_score) || Number.isFinite(assessment.rul?.hours))) {
      setLastRiskAssessment(previous => {
        const prior = previous?.sessionId === diagnosticTelemetry.session_id ? previous.assessment : {};
        return { sessionId: diagnosticTelemetry.session_id, assessment: {
          ...assessment,
          risk_fusion: Number.isFinite(assessment.risk_fusion?.engine_risk_score)
            ? assessment.risk_fusion : prior.risk_fusion,
          rul: Number.isFinite(assessment.rul?.hours) ? assessment.rul : prior.rul
        } };
      });
    }
  }, [matched, assessment, diagnosticTelemetry.session_id]);
  const hasPreviousRisk = hasSessionCache(lastRiskAssessment, diagnosticTelemetry.session_id);
  const previousRisk = !matched && hasPreviousRisk;
  const displayedRiskAssessment = matched ? assessment : previousRisk ? lastRiskAssessment.assessment : null;
  const engineHealthState = displayedRiskAssessment?.engine_state ?? displayedRiskAssessment?.engine_health_status;
  const previousRiskScore = matched && !Number.isFinite(assessment.risk_fusion?.engine_risk_score) && hasPreviousRisk;
  const previousRul = matched && !Number.isFinite(assessment.rul?.hours) && hasPreviousRisk;
  const advisoryKey = layer6Current && layer6.black_box?.predicted_class
    ? `${diagnosticTelemetry.session_id}:${diagnosticsPacket.control_revision}:${layer6.black_box.predicted_class}` : null;

  async function generateMaintenanceAdvisory() {
    if (!advisoryKey || advisoryBusy) return;
    setAdvisoryBusy(true);
    setAdvisoryError('');
    try {
      const advisory = await request('/maintenance-advisory', {
        edge_ml: {fault_class: ml.predicted_class, confidence: ml.confidence, anomaly_score: iforest.anomaly_score},
        risk_rul: {
          engine_health_percent: engineHealthPercent,
          health_state: engineHealthState,
          engine_risk_score: risk.engine_risk_score,
          subsystem_risk: risk.subsystem_risk,
          remaining_useful_life_hours: rulDisplay.hours,
          rul_interval_hours: rulDisplay.interval,
          recommended_action: displayedRiskAssessment?.recommended_action,
        },
      });
      setMaintenanceAdvice({ advisory });
    } catch (error) {
      setAdvisoryError(error.message);
    } finally {
      setAdvisoryBusy(false);
    }
  }

  useEffect(() => {
    if (viewerReady && viewer.current) {
      viewer.current.contentWindow?.postMessage({ type: 'telemetry', payload: telemetry, live: fresh }, location.origin);
    }
  }, [packet, viewerReady, fresh]);

  async function send(path, body) {
    setBusy(true);
    setNotice('');
    try {
      const result = await request(path, body);
      setRevision(result.control_revision ?? null);
      setNotice(`Controls accepted · Rev ${result.control_revision ?? 'OK'}`);
      return result;
    } catch (e) {
      setNotice(`Control failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  const entries = Object.entries(frame.values || {}).filter(([channel]) =>
    channel.toLowerCase().includes(search.toLowerCase())
  );
  const previousLayer6 = !layer6Current && hasSessionCache(lastLayer6Assessment, diagnosticTelemetry.session_id);
  const rawLayer6 = layer6Current ? layer6.black_box : previousLayer6 ? lastLayer6Assessment.assessment.black_box : {};
  const layer6Status = layer6Current ? 'CURRENT' : previousLayer6 ? 'PREVIOUS' : 'PENDING';
  const ml = {
    predicted_class: rawLayer6.predicted_class,
    class_probabilities: rawLayer6.class_probabilities,
    confidence: rawLayer6.confidence,
    feature_attributions: rawLayer6.shap_top_features,
    subsystem_attributions: rawLayer6.shap_subsystem,
    model_version: rawLayer6.model_version
  };
  const iforest = {
    anomaly_score: rawLayer6.anomaly_score,
    anomaly_detected: rawLayer6.anomaly_detected,
    novel_degradation: rawLayer6.novel_degradation,
    model_version: rawLayer6.model_version
  };
  const shownLayer6 = layer6Current ? layer6 : previousLayer6 ? lastLayer6Assessment.assessment : {};
  const risk = previousRiskScore ? lastRiskAssessment.assessment.risk_fusion || {} : displayedRiskAssessment?.risk_fusion || {};
  const engineHealthPercent = Number.isFinite(risk.engine_risk_score)
    ? Math.max(0, Math.min(100, 100 - risk.engine_risk_score)) : null;
  const rul = previousRul ? lastRiskAssessment.assessment.rul || {} : displayedRiskAssessment?.rul || {};
  const rulState = previousRul
    ? lastRiskAssessment.assessment.engine_state ?? lastRiskAssessment.assessment.engine_health_status
    : engineHealthState;
  const rulDisplay = displayedRul(rul, rulState);
  const rulConfidencePercent = Number.isFinite(rul.confidence_level) ? format(rul.confidence_level * 100, 0) : '90';
  const rulIntervalNote = rulDisplay.interval
    ? `${rulConfidencePercent}% unadjusted model interval ${format(rulDisplay.interval[0])}–${format(rulDisplay.interval[1])} h.` : '';

  return (
    <>
      <header>
        <div className="brand" style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center'}}>
          <div className="section-eyebrow" style={{marginBottom: '0.1rem'}}>Flight Operations Workstation</div>
          <div style={{fontWeight: 900, fontSize: '1.2rem'}}>MALE UAV Digital Twin Operations Console</div>
        </div>
        <div className="header-right" style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <Badge tone={fresh ? 'good' : 'bad'}>
            {fresh ? 'SIMULATED TELEMETRY STREAMING' : 'OFFLINE / STALE'}
          </Badge>
          
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: '1.2' }}>
            <div style={{ color: '#94a3b8', fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px' }}>
              Telemetry Clock (UTC)
            </div>
            <div className="mono" style={{ fontSize: '16px', fontWeight: 'bold', color: '#334155' }}>
              {Number.isFinite(telemetry.timestamp) 
                ? (telemetry.timestamp > 1e8 
                    ? new Date(telemetry.timestamp * 1000).toISOString().substring(11, 23) 
                    : new Date(telemetry.timestamp * 1000).toISOString().substring(11, 23))
                : '00:00:00.000'}
            </div>
          </div>

          <div style={{
            display: 'flex',
            border: '1px solid #cbd5e1',
            borderRadius: '6px',
            padding: '3px',
            background: '#f1f5f9',
            gap: '2px'
          }}>
            <div className="status-badge badge-live" style={{
              padding: '4px 16px',
              fontSize: '13px',
              fontWeight: 'bold',
              borderRadius: '4px',
              transition: 'all 0.2s'
            }}>
              Live
            </div>
            <div className="status-badge badge-replay" style={{
              padding: '4px 16px',
              fontSize: '13px',
              fontWeight: 'bold',
              borderRadius: '4px',
              transition: 'all 0.2s'
            }}>
              Replay
            </div>
          </div>
        </div>
      </header>

      <div className="workspace">
        {/* Constant Left Sidebar: Simulation Control Deck */}
        <aside>
          <label htmlFor="env-profile-select">Atmospheric Chamber Profile</label>
          <select
            id="env-profile-select"
            value={environment.profile}
            onChange={async e => {
              const prof = e.target.value;
              setEnvironment(prev => ({ ...prev, profile: prof }));
              const r = await send('/controls/environment', { profile: prof, altitude: 0, ambient_temp: 25 });
              if (r) {
                setEnvironment({
                  profile: r.atmospheric_profile,
                  altitude: r.altitude,
                  ambient_temp: r.ambient_temp
                });
              }
            }}
          >
            {profiles.map(p => (
              <option key={p} value={p}>{label(p)}</option>
            ))}
          </select>

          <hr />

          <label htmlFor="mission-replay-select">Recorded Mission Flight Replay</label>
          <select
            id="mission-replay-select"
            value={mission}
            onChange={e => setMission(e.target.value)}
          >
            {missions.map(m => {
              const id = typeof m === 'string' ? m : m.name;
              return <option key={id} value={id}>{id}</option>;
            })}
          </select>

          <div className="button-row">
            <button
              type="button"
              disabled={!mission || busy}
              onClick={() => send('/replay/start', { mission })}
            >
              Start Replay
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => send('/replay/stop', {})}
            >
              Return Live
            </button>
          </div>

          <button
            type="button"
            disabled={busy}
            style={{ marginTop: '12px', color: '#991b1b', borderColor: '#fecaca', marginBottom: '16px' }}
            onClick={async () => {
              if (await send('/reset', {})) setHistory([]);
            }}
          >
            Reset Simulation Session
          </button>

          <hr />

          <h2>Simulation Control Deck</h2>
          <p className="muted">Real-time telemetry override injection and environmental chamber.</p>

          <label className="toggle">
            <input
              type="checkbox"
              checked={manual}
              onChange={e => setManual(e.target.checked)}
            />
            Manual Override Mode
            <span>{manual ? 'ACTIVE' : 'OFF'}</span>
          </label>

          <label htmlFor="fault-select">Injected Fault Scenario / Operating Mode</label>
          <select
            id="fault-select"
            value={fault}
            onChange={async e => {
              const selectedFault = e.target.value;
              setFault(selectedFault);
              if (selectedFault === 'NONE') {
                setManual(false);
                setFlags({});
                await send('/controls/manual', {
                  manual_override: false,
                  injected_fault: 'NONE',
                  override_flags: {},
                  override_values: {}
                });
              } else {
                setManual(true);
                await send('/controls/manual', {
                  manual_override: true,
                  injected_fault: selectedFault,
                  override_flags: flags,
                  override_values: values
                });
              }
            }}
          >
            {faults.map(f => (
              <option key={f} value={f}>
                {f === 'NONE' ? 'STANDARD MODE (IDEAL STATE)' : label(f)}
              </option>
            ))}
          </select>

          <h3>Sensor Overrides</h3>
          <p className="muted">Checked sensors are pinned to manual values. Leave them unchecked to derive readings from RPM, throttle, ambient temperature, and altitude.</p>
          <div className="overrides">
            {Object.entries(catalog).map(([key, spec]) => (
              <div className="override" key={key}>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={!!flags[key]}
                    disabled={!manual}
                    onChange={e => setFlags({ ...flags, [key]: e.target.checked })}
                  />
                  {names[key] || label(key)}
                  <span>{spec.unit}</span>
                </label>
                <div className="input-pair">
                  <input
                    type="range"
                    disabled={!manual || !flags[key]}
                    min={spec.minimum}
                    max={spec.maximum}
                    step={(spec.maximum - spec.minimum) <= 2 ? 0.01 : (spec.maximum - spec.minimum) <= 100 ? 0.01 : 0.1}
                    value={values[key] ?? spec.default}
                    onChange={e => setValues({ ...values, [key]: Number(e.target.value) })}
                  />
                  <input
                    type="number"
                    disabled={!manual || !flags[key]}
                    min={spec.minimum}
                    max={spec.maximum}
                    step="any"
                    value={values[key] ?? spec.default}
                    onChange={e => setValues({ ...values, [key]: Number(e.target.value) })}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="button-row">
            <button
              type="button"
              className="primary"
              disabled={busy || !online}
              onClick={() => send('/controls/manual', {
                manual_override: manual,
                injected_fault: fault,
                override_flags: flags,
                override_values: values
              })}
            >
              Apply Controls
            </button>
            <button
              type="button"
              disabled={busy || !online}
              onClick={async () => {
                const r = await send('/controls/manual', {
                  manual_override: false,
                  injected_fault: 'NONE',
                  override_flags: {},
                  override_values: {}
                });
                if (r) {
                  setManual(false);
                  setFlags({});
                  setFault('NONE');
                }
              }}
            >
              Restore Auto
            </button>
          </div>


          {notice && <div className="notice">{notice}</div>}
        </aside>

        {/* Right Main Container */}
        <main>
          <div className="page-heading">
            <div className="brand" style={{display: 'flex', alignItems: 'center'}}>
              <span className="brand-icon" style={{fontSize: '2rem'}}>⌁</span>
              <div>
                <div style={{fontSize: '2rem', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '2px'}}>MALE UAV</div>
                <small style={{fontSize: '1rem', letterSpacing: '1px'}}>Aero-Piston Engine Digital Twin</small>
              </div>
            </div>
            <div>
              <Badge tone={online ? 'good' : 'bad'}>
                {online ? 'SYSTEM ONLINE' : 'OFFLINE'}
              </Badge>
            </div>
          </div>

          {!online && (
            <div className="alert bad">
              <strong>Telemetry Link Disconnected:</strong> {error || 'Awaiting connection to local API...'}. Reconnecting automatically.
            </div>
          )}
          {pending && (
            <div className="alert">
              <strong>Assessment Sync in Progress:</strong> Awaiting full pipeline convergence for control revision {packet?.control_revision ?? '—'}. Previous risk and RUL, when available, are labeled below.
            </div>
          )}

          {/* 7-Layer Architecture Overview Bar */}
          <div className="pipeline-strip">
            {[
              ['01', 'Sensor Gen'],
              ['02', 'Synchronization'],
              ['03', 'Physics PINNs'],
              ['04', 'Residuals EKF'],
              ['05', 'Feature Windows'],
              ['06', 'AI Classifiers'],
              ['07', 'Risk & RUL']
            ].map(([num, title]) => (
              <div key={num}>
                <b>L{num}</b>
                <span>{title}</span>
              </div>
            ))}
          </div>

          {/* 3D Engine Visualizer */}
          <div className="engine panel">
            <iframe
              ref={viewer}
              onLoad={() => setViewerReady(true)}
              title="Three.js engine visualizer"
              src="/engine/index.html?v=20260927_04"
            />
          </div>

          {/* Metric Readouts Grid */}
          <div className="metric-grid">
            <Metric name="Engine Speed" value={format(telemetry.rpm, 0)} unit="RPM" note="Target: 5,200" />
            <Metric name="Exhaust Temp (EGT)" value={format(telemetry.egt, 1)} unit="°C" note="Limit: 850 °C" />
            <Metric name="Cylinder Head (CHT)" value={format(telemetry.cht, 1)} unit="°C" note="Limit: 220 °C" />
            <Metric name="Oil Pressure" value={format(telemetry.oil_press, 1)} unit="PSI" note="Nominal: 45–65" />
            <Metric name="Engine Vibration" value={format(telemetry.vibration, 3)} unit="g" note="Spectral RMS" />
          </div>

          {/* Synchronized Sensor Data */}
          <div className="panel">
            <div className="panel-title">
              <h2>Synchronized Sensor Data</h2>
              <input
                type="text"
                placeholder="Filter channels..."
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Sensor Channel</th>
                    <th>Synchronized Value</th>
                    <th>Channel Validity</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(([k, v]) => (
                    <tr key={k}>
                      <td className="mono">{SENSOR_CHANNEL_LABELS[k] || k}</td>
                      <td className="mono">
                        {frame.channel_valid?.[k] === true ? format(v, 3) : 'Unavailable'}
                      </td>
                      <td>
                        <Badge tone={frame.channel_valid?.[k] === true ? 'good' : 'bad'}>
                          {frame.channel_valid?.[k] === true ? 'VALID' : 'INVALID'}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                  {!entries.length && (
                    <tr>
                      <td colSpan={3} className="empty">No channels match filter query.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Two-Column Grid: Physics Predictions & AI Diagnostics */}
          <div className="two-col">
            <div className="panel">
              <div className="panel-title">
                <h2>Physics PINN & EKF Residuals</h2>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Measurement</th>
                      <th>Actual</th>
                      <th>Expected</th>
                      <th>Residual</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['egt', 'cht', 'oil_temp', 'map', 'oil_press', 'fuel_flow', 'iat', 'vibration'].map(k => (
                      <tr key={k}>
                        <td>{names[k] || k}</td>
                        <td className="mono">{format(physicsRows[k].actual)}</td>
                        <td className="mono">{format(physicsRows[k].expected)}</td>
                        <td className="mono">{format(physicsRows[k].residual)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel">
              <div className="panel-title">
                <h2>Edge ML Diagnostics</h2>
              </div>
              <div className="model-status">
                <div>
                  <small>Predicted Fault Class</small>
                  <strong>{layer6Status === 'PENDING' ? 'Waiting for Layer 6' : label(ml.predicted_class)}</strong>
                </div>
                <div>
                  <small>Confidence</small>
                  <strong>{ml.confidence == null ? 'Waiting' : `${format(ml.confidence * 100)} %`}</strong>
                </div>
                <div>
                  <small>Anomaly Score</small>
                  <strong>{iforest.anomaly_score == null ? 'Waiting' : format(iforest.anomaly_score, 4)}</strong>
                </div>
              </div>
              {layer6Status === 'PENDING' && (layer6.classification_unavailable_reason || layer6.decision_eligibility_reason) &&
                <p className="muted">{layer6.classification_unavailable_reason || layer6.decision_eligibility_reason}</p>}
            </div>
          </div>

          {/* LAYER 07 · Engine Health & Prognostics */}
          <div className="panel">
            <div className="panel-title">
              <h2>Engine Health & RUL Prognostics</h2>
              <Badge tone={engineHealthState === 'critical' ? 'bad' : engineHealthState === 'normal' ? 'good' : 'neutral'}>
                {matched ? label(engineHealthState ?? 'PENDING') : previousRisk ? 'PREVIOUS' : 'PENDING'}
              </Badge>
            </div>
            {!matched && displayedRiskAssessment &&
              <p className="muted">Showing the previous risk/RUL assessment while Layer 7 catches up to the current telemetry and control revision.</p>}
            <div className="metric-grid">
              <Metric name="Engine Health" value={engineHealthPercent == null ? 'Waiting' : format(engineHealthPercent)} unit="%" />
              <Metric name="Health State" value={engineHealthState && engineHealthState !== 'unknown' ? label(engineHealthState) : displayedRiskAssessment ? 'Unknown' : 'Waiting'} />
              <Metric name="Remaining Useful Life (RUL)" value={rulDisplay.hours == null ? displayedRiskAssessment ? 'Not estimable' : 'Waiting' : format(rulDisplay.hours)} unit="h" />
              <Metric name="Action Advisory" value={displayedRiskAssessment?.recommended_action ? label(displayedRiskAssessment.recommended_action) : 'Waiting'} />
            </div>
            <div className="risk-bars">
              {Object.entries(risk.subsystem_risk || {}).map(([k, v]) => (
                <div key={k}>
                  <span>{label(k)}</span>
                  <progress max="100" value={Math.min(100, Math.max(0, Number(v) || 0))} />
                  <b>{Number.isFinite(v) ? format(v) : 'Waiting'}</b>
                </div>
              ))}
            </div>
            {matched && Number(risk.engine_risk_confidence) === 0 &&
              <p className="muted">No usable physical residual or NIS evidence reached Layer 7 for this assessment.</p>}
            <div className="maintenance-advisory">
              <h3>XAI maintenance advisory</h3>
              <button className="primary" type="button" onClick={generateMaintenanceAdvisory} disabled={!advisoryKey || advisoryBusy}>
                {advisoryBusy ? 'Generating advisory…' : 'Generate Maintenance Advisory'}
              </button>
              {advisoryError && <p className="muted">{advisoryError}</p>}
              {maintenanceAdvice?.advisory?.text &&
                <p style={{whiteSpace: 'pre-wrap'}}>{maintenanceAdvice.advisory.text}</p>}

            </div>

          </div>

          {/* Live Telemetry History Oscilloscope (Tab 04) */}
          <TrendsPanel history={history} />

          <footer>
            <div>MALE UAV DIGITAL TWIN WORKSTATION</div>
            <div>FastAPI Real-time Port 8000</div>
          </footer>
        </main>
      </div>
    </>
  );
}

createRoot(document.getElementById('root')).render(<App />);
