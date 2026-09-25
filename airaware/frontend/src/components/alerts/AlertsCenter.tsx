import { AlertCircle, Bell, CheckCircle2, X } from 'lucide-react'
import { useEffect } from 'react'
import { useAlerts } from '../../hooks/useAlerts'
import AlertCard from './AlertCard'

export default function AlertsCenter() {
  const { activeAlerts, resolvedAlerts, status, error, isOpen, tab, setTab, closeAlerts } = useAlerts()
  const visible = tab === 'active' ? activeAlerts : resolvedAlerts

  useEffect(() => {
    if (!isOpen) return
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') closeAlerts() }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [closeAlerts, isOpen])

  if (!isOpen) return null
  return <>
    <button type="button" className="alerts-scrim" onClick={closeAlerts} aria-label="Close Alerts Center" />
    <aside className="alerts-center" role="dialog" aria-modal="true" aria-labelledby="alerts-center-title">
      <div className="alerts-center-head">
        <div><span><Bell />Alert intelligence</span><h2 id="alerts-center-title">Alerts</h2><p>Development preview · not live monitoring</p></div>
        <button type="button" onClick={closeAlerts} aria-label="Close Alerts Center" autoFocus><X /></button>
      </div>
      <div className="alerts-tabs" role="tablist" aria-label="Alert status">
        <button type="button" role="tab" aria-selected={tab === 'active'} className={tab === 'active' ? 'active' : ''} onClick={() => setTab('active')}>Active <b>{activeAlerts.length}</b></button>
        <button type="button" role="tab" aria-selected={tab === 'resolved'} className={tab === 'resolved' ? 'active' : ''} onClick={() => setTab('resolved')}>Resolved <b>{resolvedAlerts.length}</b></button>
      </div>
      <div className="alerts-center-content">
        {status === 'loading' && <div className="alerts-state"><span className="alerts-loader" /><strong>Loading alerts</strong><p>Preparing the development alert feed.</p></div>}
        {status === 'error' && <div className="alerts-state error"><AlertCircle /><strong>Alerts unavailable</strong><p>{error || 'The alert feed could not be loaded.'}</p></div>}
        {status === 'ready' && visible.length === 0 && <div className="alerts-state clear"><CheckCircle2 /><strong>All clear</strong><p>{tab === 'active' ? 'No active alerts detected.' : 'No resolved alerts to show.'}</p></div>}
        {status === 'ready' && visible.map(alert => <AlertCard key={alert.id} alert={alert} />)}
      </div>
    </aside>
  </>
}
