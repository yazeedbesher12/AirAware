import { BellRing, ChevronRight } from 'lucide-react'
import { useAlerts } from '../../hooks/useAlerts'

export default function AlertsSummary() {
  const { activeAlerts, status, openAlerts } = useAlerts()
  const critical = activeAlerts.filter(alert => alert.severity === 'critical').length
  const warning = activeAlerts.filter(alert => alert.severity === 'warning').length
  return <button type="button" className="alerts-summary" onClick={() => openAlerts('active')} disabled={status === 'loading'} aria-label={`Open Alerts Center. ${activeAlerts.length} active alerts`}>
    <span className="alerts-summary-icon"><BellRing /></span>
    <span><small>Alerts Center</small><b>{status === 'loading' ? 'Loading…' : `${activeAlerts.length} Active Alert${activeAlerts.length === 1 ? '' : 's'}`}</b><em>{critical} Critical · {warning} Warning</em></span>
    <ChevronRight />
  </button>
}
