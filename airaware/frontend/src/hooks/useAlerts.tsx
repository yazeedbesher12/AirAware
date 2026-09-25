import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { getAlerts } from '../services/alerts'
import type { AirAwareAlert } from '../types/alerts'

type AlertTab = 'active' | 'resolved'

interface AlertsContextValue {
  alerts: AirAwareAlert[]
  activeAlerts: AirAwareAlert[]
  resolvedAlerts: AirAwareAlert[]
  status: 'loading' | 'ready' | 'error'
  error: string | null
  isOpen: boolean
  tab: AlertTab
  openAlerts: (tab?: AlertTab) => void
  closeAlerts: () => void
  setTab: (tab: AlertTab) => void
}

const AlertsContext = createContext<AlertsContextValue | null>(null)

export function AlertsProvider({ children }: { children: React.ReactNode }) {
  const [alerts, setAlerts] = useState<AirAwareAlert[]>([])
  const [status, setStatus] = useState<AlertsContextValue['status']>('loading')
  const [error, setError] = useState<string | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [tab, setTab] = useState<AlertTab>('active')

  useEffect(() => {
    let current = true
    getAlerts().then(data => {
      if (!current) return
      setAlerts(data)
      setStatus('ready')
    }).catch(reason => {
      if (!current) return
      setError(reason instanceof Error ? reason.message : 'Alerts could not be loaded.')
      setStatus('error')
    })
    return () => { current = false }
  }, [])

  const value = useMemo<AlertsContextValue>(() => ({
    alerts,
    activeAlerts: alerts.filter(alert => alert.status === 'active'),
    resolvedAlerts: alerts.filter(alert => alert.status === 'resolved'),
    status,
    error,
    isOpen,
    tab,
    openAlerts: (nextTab = 'active') => { setTab(nextTab); setIsOpen(true) },
    closeAlerts: () => setIsOpen(false),
    setTab,
  }), [alerts, status, error, isOpen, tab])

  return <AlertsContext.Provider value={value}>{children}</AlertsContext.Provider>
}

export function useAlerts() {
  const context = useContext(AlertsContext)
  if (!context) throw new Error('useAlerts must be used inside AlertsProvider')
  return context
}
