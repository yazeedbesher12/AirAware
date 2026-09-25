import type { AirAwareAlert } from '../types/alerts'

// Development-only examples shaped exactly like the future AlertEngine output.
// Historical health examples remain explicitly tagged as demo data in the UI.
export const mockAlerts: AirAwareAlert[] = [
  {
    id: 'alert_demo_pm25_3h', dedup_key: '1:high_pm_risk:3h:pm25', sensor_id: 1,
    site: 'Tulkarem Municipality', city: 'Tulkarem', type: 'high_pm_risk', category: 'forecast',
    severity: 'warning', priority: 'medium', status: 'active', title: 'Elevated PM2.5 risk expected',
    message: 'Elevated PM2.5 risk is forecast in 3 hours.', created_at: '2026-09-02T09:30:00Z',
    updated_at: '2026-09-02T09:45:00Z', resolved_at: null, horizon: '3h', channel: 'pm25',
    source: 'forecast_engine', evidence: { forecast_pm25: 17.355, safety_upper_estimate: 23.499, safety_margin: 6.144, risk_level: 'Elevated', horizon: '3h' },
    alert_recommended: true, notification_recommended: false, is_early_warning: true, demo: true,
  },
  {
    id: 'alert_demo_pm25_6h', dedup_key: '1:high_pm_risk:6h:pm25', sensor_id: 1,
    site: 'Tulkarem Municipality', city: 'Tulkarem', type: 'high_pm_risk', category: 'forecast',
    severity: 'warning', priority: 'high', status: 'active', title: 'High PM2.5 risk expected',
    message: 'High PM2.5 risk is forecast in 6 hours.', created_at: '2026-09-02T09:30:00Z',
    updated_at: '2026-09-02T09:45:00Z', resolved_at: null, horizon: '6h', channel: 'pm25',
    source: 'forecast_engine', evidence: { forecast_pm25: 23.717, safety_upper_estimate: 31.538, safety_margin: 7.821, risk_level: 'High', horizon: '6h' },
    alert_recommended: true, notification_recommended: true, is_early_warning: true, demo: true,
  },
  {
    id: 'alert_demo_who_3h', dedup_key: '1:who_exceedance:3h:pm25', sensor_id: 1,
    site: 'Tulkarem Municipality', city: 'Tulkarem', type: 'who_exceedance', category: 'health_guideline',
    severity: 'warning', priority: 'medium', status: 'active', title: 'WHO PM2.5 24h guideline exceedance',
    message: 'Predicted PM2.5 24-hour average is above the WHO guideline.', created_at: '2026-09-02T09:30:00Z',
    updated_at: '2026-09-02T09:45:00Z', resolved_at: null, horizon: '3h', channel: 'pm25',
    source: 'forecast_engine', evidence: { predicted_24h_average: 17.59, guideline: 15, ratio_to_guideline: 1.17, horizon: '3h' },
    alert_recommended: true, notification_recommended: false, is_early_warning: true, demo: true,
  },
  {
    id: 'alert_demo_offline_5', dedup_key: '5:sensor_offline', sensor_id: 5,
    site: 'Hisham Hijjawi College of Technology', city: 'Nablus', type: 'sensor_offline', category: 'sensor_health',
    severity: 'critical', priority: 'urgent', status: 'active', title: 'Sensor Offline',
    message: 'Sensor 5 is offline.', created_at: '2026-09-02T08:20:00Z', updated_at: '2026-09-02T09:45:00Z',
    resolved_at: null, horizon: null, channel: null, source: 'sensor_health_engine',
    evidence: { last_reading_at: '2026-09-02T08:20:00Z', minutes_since_last_reading: 85, site: 'Hisham Hijjawi College of Technology' },
    alert_recommended: true, notification_recommended: false, is_early_warning: false, demo: true,
  },
  {
    id: 'alert_demo_drift_2', dedup_key: '2:possible_drift:pm25', sensor_id: 2,
    site: 'ANNU New Campus', city: 'Nablus', type: 'possible_drift', category: 'sensor_health',
    severity: 'warning', priority: 'high', status: 'active', title: 'Possible Sensor Drift',
    message: 'Possible Drift detected in the PM2.5 channel.', created_at: '2026-09-02T08:45:00Z',
    updated_at: '2026-09-02T09:45:00Z', resolved_at: null, horizon: null, channel: 'pm25',
    source: 'sensor_health_engine', evidence: { median_shift: 15.9, recent_samples: 96, baseline_samples: 192 },
    alert_recommended: true, notification_recommended: false, is_early_warning: false, demo: true,
  },
  {
    id: 'alert_demo_suspicious_1', dedup_key: '1:suspicious_reading:o3', sensor_id: 1,
    site: 'Tulkarem Municipality', city: 'Tulkarem', type: 'suspicious_reading', category: 'sensor_health',
    severity: 'critical', priority: 'urgent', status: 'resolved', title: 'Suspicious Sensor Reading',
    message: 'Historical demo: a negative O₃ reading was detected and later cleared.', created_at: '2026-08-16T16:15:00Z',
    updated_at: '2026-08-16T17:00:00Z', resolved_at: '2026-08-16T17:00:00Z', horizon: null, channel: 'o3',
    source: 'sensor_health_engine', evidence: { value: -2.32, lower_bound: 0, reason: 'Outside broad physical plausibility bounds.' },
    alert_recommended: true, notification_recommended: false, is_early_warning: false, demo: true,
  },
]
