export type AlertSeverity = 'info' | 'warning' | 'critical'
export type AlertPriority = 'low' | 'medium' | 'high' | 'urgent'
export type AlertStatus = 'active' | 'resolved'
export type AlertCategory = 'forecast' | 'health_guideline' | 'sensor_health'
export type AlertType = 'high_pm_risk' | 'who_exceedance' | 'sensor_offline' | 'possible_drift' | 'suspicious_reading'

export interface AirAwareAlert {
  id: string
  dedup_key: string
  sensor_id: number
  site: string
  city: string
  type: AlertType
  category: AlertCategory
  severity: AlertSeverity
  priority: AlertPriority
  status: AlertStatus
  title: string
  message: string
  created_at: string
  updated_at: string
  resolved_at: string | null
  horizon: '1h' | '3h' | '6h' | null
  channel: string | null
  source: 'forecast_engine' | 'sensor_health_engine'
  evidence: Record<string, string | number | boolean | null>
  alert_recommended: boolean
  notification_recommended: boolean
  is_early_warning: boolean
  demo: true
}
