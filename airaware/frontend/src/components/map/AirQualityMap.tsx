import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, Compass, LocateFixed, Radio, Sparkles } from 'lucide-react'
import { AttributionControl, LngLatBounds, Map as MapLibreMap, NavigationControl, type StyleSpecification } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './AirQualityMap.css'
import { useSensors } from '../../hooks/useSensors'
import { getSensorSnapshot } from '../../services/sensors'
import type { RiskLevel, TimelineMode } from '../../types/sensors'
import MapLegend from './MapLegend'
import SensorDetails from './SensorDetails'
import SensorMarker from './SensorMarker'

const TIMELINES: Array<{ value: TimelineMode; label: string }> = [
  { value: 'current', label: 'Current' },
  { value: '1h', label: '+1h' },
  { value: '3h', label: '+3h' },
  { value: '6h', label: '+6h' },
]
const FILTERS: Array<'All' | RiskLevel> = ['All', 'Normal', 'Elevated', 'High']

const PALESTINE_BASEMAP: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [{
    id: 'palestine-basemap',
    type: 'raster',
    source: 'osm',
    paint: {
      'raster-saturation': -0.62,
      'raster-contrast': -0.08,
      'raster-brightness-min': 0.12,
      'raster-brightness-max': 0.95,
    },
  }],
}

export default function AirQualityMap() {
  const { sensors, status, error } = useSensors()
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [timeline, setTimeline] = useState<TimelineMode>('current')
  const [riskFilter, setRiskFilter] = useState<'All' | RiskLevel>('All')
  const [selectedId, setSelectedId] = useState<number | null>(1)

  const sensorViews = useMemo(() => sensors.map(sensor => ({ sensor, snapshot: getSensorSnapshot(sensor, timeline) })), [sensors, timeline])
  const visibleSensors = useMemo(() => sensorViews.filter(({ snapshot }) => riskFilter === 'All' || snapshot.highPmRisk === riskFilter), [riskFilter, sensorViews])
  const selectedView = sensorViews.find(({ sensor }) => sensor.id === selectedId) ?? null
  const summary = useMemo(() => ({
    total: sensors.length,
    online: sensors.filter(sensor => sensor.online).length,
    elevated: sensorViews.filter(({ snapshot }) => snapshot.highPmRisk === 'Elevated').length,
    high: sensorViews.filter(({ snapshot }) => snapshot.highPmRisk === 'High').length,
  }), [sensorViews, sensors])

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return
    const map = new MapLibreMap({
      container: mapContainerRef.current,
      style: PALESTINE_BASEMAP,
      center: [35.14, 32.26],
      zoom: 10.2,
      minZoom: 8,
      maxZoom: 16,
      attributionControl: false,
    })
    map.addControl(new NavigationControl({ showCompass: false }), 'bottom-left')
    map.addControl(new AttributionControl({ compact: true }), 'bottom-right')
    map.on('load', () => setMapReady(true))
    mapRef.current = map

    const observer = new ResizeObserver(() => map.resize())
    observer.observe(mapContainerRef.current)
    return () => {
      observer.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  const fitNetwork = (panelOpen = Boolean(selectedView)) => {
    const map = mapRef.current
    if (!map || !sensors.length) return
    const bounds = sensors.reduce((next, sensor) => next.extend([sensor.coordinates.longitude, sensor.coordinates.latitude]), new LngLatBounds())
    const wide = map.getContainer().clientWidth > 900
    map.fitBounds(bounds, {
      padding: wide ? { top: 110, right: panelOpen ? 430 : 90, bottom: 95, left: 90 } : { top: 130, right: 65, bottom: 100, left: 65 },
      maxZoom: 11.15,
      duration: 900,
    })
  }

  useEffect(() => {
    if (mapReady && sensors.length) fitNetwork()
    // Fit once when the development sensor source becomes available.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapReady, sensors.length])

  useEffect(() => {
    if (riskFilter === 'All' || selectedView?.snapshot.highPmRisk === riskFilter) return
    setSelectedId(visibleSensors[0]?.sensor.id ?? null)
  }, [riskFilter, selectedView, visibleSensors])

  const selectSensor = (sensorId: number) => {
    const target = sensors.find(sensor => sensor.id === sensorId)
    if (!target) return
    setSelectedId(sensorId)
    mapRef.current?.easeTo({
      center: [target.coordinates.longitude, target.coordinates.latitude],
      zoom: Math.max(mapRef.current.getZoom(), 11.4),
      offset: mapRef.current.getContainer().clientWidth > 900 ? [-190, 0] : [0, -70],
      duration: 850,
    })
  }

  return <section className="air-quality-map-section" aria-labelledby="air-map-heading">
    <div className="map-command-head">
      <div className="map-intro">
        <span className="map-eyebrow"><Sparkles /> Palestine environmental intelligence</span>
        <h2 id="air-map-heading">Air Quality Map</h2>
        <p>Monitor current conditions and explore how PM2.5 may change across AirAware’s Palestinian sensor network.</p>
      </div>
      <div className="map-summary" aria-label="Sensor network summary">
        <span><b>{summary.total}</b><small>Sensors</small></span>
        <span><b>{summary.online}</b><small>Online</small></span>
        <span className="summary-elevated"><b>{summary.elevated}</b><small>Elevated risk</small></span>
        <span className="summary-high"><b>{summary.high}</b><small>High risk</small></span>
      </div>
    </div>

    <div className="map-toolbar">
      <div className="timeline-control" role="group" aria-label="Forecast timeline">
        <span>Map mode</span>
        <div>{TIMELINES.map(option => <button type="button" key={option.value} className={timeline === option.value ? 'active' : ''} onClick={() => setTimeline(option.value)} aria-pressed={timeline === option.value}>{option.label}</button>)}</div>
      </div>
      <div className="risk-filters" role="group" aria-label="Filter by AirAware High-PM Risk">
        <span>AirAware Risk</span>
        <div>{FILTERS.map(filter => <button type="button" key={filter} className={`${riskFilter === filter ? 'active ' : ''}filter-${filter.toLowerCase()}`} onClick={() => setRiskFilter(filter)} aria-pressed={riskFilter === filter}>{filter}</button>)}</div>
      </div>
      <div className="map-mode-note"><Radio /><span>{timeline === 'current' ? 'Current readings' : `${timeline} forecast`}<small>{timeline === 'current' ? 'Latest sensor conditions' : `Forecast state · ${timeline.replace('h', ' hours')} ahead`}</small></span></div>
    </div>

    <div className={`map-stage${selectedView ? ' has-details' : ''}`}>
      <div className="map-canvas-wrap">
        <div ref={mapContainerRef} className="map-canvas" aria-label="Map of AirAware sensors in Tulkarem and Nablus, Palestine" />
        <div className="map-context"><Compass /><span>Palestine sensor network<small>Tulkarem · Nablus</small></span></div>
        <button className="fit-network" type="button" onClick={() => fitNetwork()}><LocateFixed />Fit sensor network</button>
        <MapLegend />
        {status === 'loading' && <div className="map-data-loading"><span />Preparing sensor layer…</div>}
        {status === 'ready' && !sensors.length && <div className="map-empty-state"><AlertCircle /><strong>No sensor data</strong><span>Sensor locations will appear here when data is available.</span></div>}
        {status === 'error' && <div className="map-empty-state error"><AlertCircle /><strong>Sensor data unavailable</strong><span>{error}</span></div>}
        {mapReady && mapRef.current && visibleSensors.map(({ sensor, snapshot }) => <SensorMarker key={sensor.id} map={mapRef.current!} sensor={sensor} snapshot={snapshot} timeline={timeline} selected={selectedId === sensor.id} onSelect={() => selectSensor(sensor.id)} />)}
        {mapReady && sensors.length > 0 && visibleSensors.length === 0 && <div className="no-risk-results">No sensors match this risk level in the selected timeline.</div>}
      </div>
      {selectedView && <SensorDetails sensor={selectedView.sensor} snapshot={selectedView.snapshot} timeline={timeline} onClose={() => { setSelectedId(null); requestAnimationFrame(() => fitNetwork(false)) }} />}
    </div>
  </section>
}
