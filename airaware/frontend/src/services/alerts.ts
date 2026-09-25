import { mockAlerts } from '../data/alertsMock'
import type { AirAwareAlert } from '../types/alerts'

export function adaptAlerts(source: AirAwareAlert[]): AirAwareAlert[] {
  return source.filter(alert => alert && alert.id && alert.dedup_key && alert.title && alert.message)
}

export async function getAlerts(): Promise<AirAwareAlert[]> {
  // Future integration point: replace the development source with GET /api/alerts.
  // The UI consumes this adapter only; no fake endpoint is created here.
  return Promise.resolve(adaptAlerts(mockAlerts))
}
