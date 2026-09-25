import type { TimelineMode } from '../../types/sensors'
import { useAlerts } from '../../hooks/useAlerts'
import AlertCard from './AlertCard'

export default function SensorAlerts({ sensorId, timeline }: { sensorId: number; timeline: TimelineMode }) {
  const { activeAlerts, openAlerts } = useAlerts()
  const alerts = activeAlerts.filter(alert => alert.sensor_id === sensorId).sort((left, right) => {
    const relevance = (alert: typeof left) => timeline === 'current'
      ? (alert.category === 'sensor_health' ? 0 : 1)
      : (alert.horizon === timeline ? 0 : alert.category === 'sensor_health' ? 1 : 2)
    return relevance(left) - relevance(right)
  })
  if (!alerts.length) return <section className="sensor-alerts empty"><div className="details-section-title"><span>Active Alerts</span><small>0</small></div><p>All clear · no active alerts for this sensor.</p></section>
  return <section className="sensor-alerts">
    <div className="details-section-title"><span>Active Alerts ({alerts.length})</span><button type="button" onClick={() => openAlerts('active')}>View all</button></div>
    <div>{alerts.map(alert => <AlertCard key={alert.id} alert={alert} compact contextual={timeline !== 'current' && alert.horizon === timeline} />)}</div>
  </section>
}
