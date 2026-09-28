export async function request(path, body, signal) {
  const response = await fetch(`/api/dashboard${path}`, {
    method: body === undefined ? 'GET' : 'POST', signal,
    headers: body === undefined ? {} : {'Content-Type':'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(()=>({}));
    throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status})`);
  }
  return response.json();
}
export function appendSample(history, sample) {
  if (!sample || !Number.isFinite(sample.timestamp)) return history;
  const last = history.at(-1);
  if (last?.session_id !== sample.session_id || sample.timestamp < last?.timestamp) return [sample];
  if (sample.timestamp === last?.timestamp) return history;
  return [...history, sample].slice(-180);
}
export function format(value, digits=1) {return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable';}
export function displayedRul(rul, engineState) {
  const hours = rul?.hours;
  if (!Number.isFinite(hours)) return {hours: null, interval: null, uplift: 0};
  const uplift = engineState === 'normal' ? 10 : 0;
  const interval = Array.isArray(rul.interval_hours) && rul.interval_hours.length === 2 &&
    rul.interval_hours.every(Number.isFinite)
    ? [...rul.interval_hours] : null;
  return {hours: hours + uplift, interval, uplift};
}
export function currentAssessment(packet, requestedRevision) {
  return !!packet?.telemetry && !packet.telemetry.control_pending &&
    packet.telemetry.assessment_control_revision === packet.control_revision &&
    (requestedRevision == null || packet.control_revision >= requestedRevision);
}
export function hasSessionCache(cache, sessionId) {
  return Boolean(sessionId && cache?.sessionId === sessionId);
}
