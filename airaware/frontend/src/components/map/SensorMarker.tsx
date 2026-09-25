import { useEffect, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { Marker, type Map as MapLibreMap } from 'maplibre-gl'
import type { SensorData, SensorSnapshot, TimelineMode } from '../../types/sensors'
import RiskBadge from './RiskBadge'
import { BellRing } from 'lucide-react'
import type { AlertSeverity } from '../../types/alerts'

const categoryClass = (category: string | null) => category ? category.toLowerCase().replaceAll(' ', '-') : 'unavailable'

export default function SensorMarker({
  map,
  sensor,
  snapshot,
  timeline,
  selected,
  alertCount,
  alertSeverity,
  onSelect,
}: {
  map: MapLibreMap
  sensor: SensorData
  snapshot: SensorSnapshot
  timeline: TimelineMode
  selected: boolean
  alertCount: number
  alertSeverity: AlertSeverity
  onSelect: () => void
}) {
  const element = useMemo(() => {
    const container = document.createElement('div')
    container.className = 'sensor-marker-host'
    return container
  }, [])

  useEffect(() => {
    const marker = new Marker({ element, anchor: 'bottom' })
      .setLngLat([sensor.coordinates.longitude, sensor.coordinates.latitude])
      .addTo(map)
    return () => { marker.remove() }
  }, [element, map, sensor.coordinates.latitude, sensor.coordinates.longitude])

  const modeLabel = timeline === 'current' ? 'Now' : `+${timeline}`

  return createPortal(
    <button
      type="button"
      className={`sensor-map-marker risk-${snapshot.highPmRisk?.toLowerCase() ?? 'unavailable'} aqi-${categoryClass(snapshot.epaCategory)}${selected ? ' selected' : ''}${sensor.online ? '' : ' offline'}`}
      onClick={onSelect}
      aria-label={`${sensor.name}, ${sensor.site}, ${sensor.city}, ${snapshot.pm25 == null ? 'PM2.5 unavailable' : `PM2.5 ${snapshot.pm25.toFixed(1)}`}, ${snapshot.highPmRisk ?? 'unavailable'} AirAware risk`}
      aria-pressed={selected}
    >
      <span className="marker-ring" />
      {alertCount > 0 && <span className={`marker-alert severity-${alertSeverity}`} aria-label={`${alertCount} active alert${alertCount === 1 ? '' : 's'}`}><BellRing />{alertCount}</span>}
      <span className="marker-card">
        <span className="marker-place"><i className={sensor.online ? 'online' : 'offline'} />S{sensor.id} · {sensor.city}<b>{modeLabel}</b></span>
        <strong>{snapshot.pm25 == null ? '—' : snapshot.pm25.toFixed(1)}</strong>
        <small>µg/m³ · PM2.5</small>
        <RiskBadge level={snapshot.highPmRisk} compact />
      </span>
      <span className="marker-stem" />
    </button>,
    element,
  )
}
