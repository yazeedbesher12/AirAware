import { AlertTriangle, CheckCircle2, Clock3, Radio, ShieldAlert, TriangleAlert } from 'lucide-react'
import type { AirAwareAlert } from '../../types/alerts'

const formatNumber = (value: unknown, digits = 1) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : null
const formatTime = (value: string) => new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short' })

function Evidence({ alert }: { alert: AirAwareAlert }) {
  if (alert.type === 'high_pm_risk') return <div className="alert-evidence">
    <span><small>PM2.5 forecast</small><b>{formatNumber(alert.evidence.forecast_pm25)} <em>µg/m³</em></b></span>
    <span><small>Safety Upper</small><b>{formatNumber(alert.evidence.safety_upper_estimate)} <em>µg/m³</em></b></span>
  </div>
  if (alert.type === 'who_exceedance') return <div className="alert-evidence">
    <span><small>Predicted 24h average</small><b>{formatNumber(alert.evidence.predicted_24h_average)} <em>µg/m³</em></b></span>
    <span><small>WHO guideline</small><b>{formatNumber(alert.evidence.guideline)} <em>µg/m³</em></b></span>
  </div>
  if (alert.type === 'sensor_offline') return <div className="alert-evidence">
    <span><small>Last reading</small><b>{alert.evidence.last_reading_at ? formatTime(String(alert.evidence.last_reading_at)) : 'Not available'}</b></span>
    <span><small>Offline duration</small><b>{formatNumber(alert.evidence.minutes_since_last_reading, 0) ?? '—'} <em>minutes</em></b></span>
  </div>
  if (alert.type === 'possible_drift') return <div className="alert-evidence">
    <span><small>Channel</small><b>{alert.channel?.toUpperCase() ?? 'Sensor data'}</b></span>
    <span><small>Classification</small><b>Possible Drift</b></span>
  </div>
  if (alert.type === 'suspicious_reading') return <div className="alert-evidence">
    <span><small>Channel</small><b>{alert.channel?.toUpperCase() ?? 'Sensor data'}</b></span>
    <span><small>Observed value</small><b>{formatNumber(alert.evidence.value, 2) ?? 'Not available'}</b></span>
  </div>
  return null
}

export default function AlertCard({ alert, compact = false, contextual = false }: { alert: AirAwareAlert; compact?: boolean; contextual?: boolean }) {
  const Icon = alert.status === 'resolved' ? CheckCircle2 : alert.severity === 'critical' ? ShieldAlert : alert.severity === 'warning' ? TriangleAlert : AlertTriangle
  return <article className={`alert-card severity-${alert.severity} status-${alert.status}${compact ? ' compact' : ''}${contextual ? ' contextual' : ''}`}>
    <div className="alert-card-icon"><Icon /></div>
    <div className="alert-card-body">
      <div className="alert-card-meta">
        <span className={`alert-severity severity-${alert.severity}`}>{alert.status === 'resolved' ? 'Resolved' : alert.severity}</span>
        {alert.is_early_warning && <span className="early-warning"><Radio />Early Warning</span>}
        {alert.horizon && <span className="alert-horizon">+{alert.horizon}</span>}
        <span className="demo-chip">Demo data</span>
      </div>
      <h4>{alert.title}</h4>
      {!compact && <><p className="alert-location">Sensor {alert.sensor_id} · {alert.site}</p><p>{alert.message}</p><Evidence alert={alert} /></>}
      <small className="alert-updated"><Clock3 />{alert.status === 'resolved' && alert.resolved_at ? `Resolved ${formatTime(alert.resolved_at)}` : `Updated ${formatTime(alert.updated_at)}`}</small>
    </div>
  </article>
}
