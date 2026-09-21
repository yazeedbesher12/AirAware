import type { SensorData, TimelineMode } from '../../types/sensors'

const WIDTH = 330
const HEIGHT = 120
const PAD_X = 24
const PAD_Y = 22

export default function ForecastMiniChart({ sensor, timeline }: { sensor: SensorData; timeline: TimelineMode }) {
  const points = [
    { id: 'current' as const, label: 'Now', value: sensor.current.pm25 },
    { id: '1h' as const, label: '1h', value: sensor.forecasts['1h'].pm25 },
    { id: '3h' as const, label: '3h', value: sensor.forecasts['3h'].pm25 },
    { id: '6h' as const, label: '6h', value: sensor.forecasts['6h'].pm25 },
  ]
  const values = points.flatMap(point => point.value == null || !Number.isFinite(point.value) ? [] : [point.value])

  if (!values.length) return <div className="forecast-mini-chart"><div className="mini-chart-title"><span>PM2.5 trajectory</span><small>Now → 6h</small></div><div className="inline-empty">Forecast unavailable</div></div>

  const min = Math.max(0, Math.min(...values) - 4)
  const max = Math.max(...values) + 4
  const plotWidth = WIDTH - PAD_X * 2
  const plotHeight = HEIGHT - PAD_Y * 2
  const coords = points.map((point, index) => ({
    ...point,
    x: PAD_X + (index * plotWidth) / (points.length - 1),
    y: point.value == null ? null : PAD_Y + ((max - point.value) / Math.max(1, max - min)) * plotHeight,
  }))
  const segments: Array<Array<{ x: number; y: number }>> = []
  coords.forEach((point, index) => {
    if (point.y == null) return
    if (index === 0 || coords[index - 1].y == null) segments.push([])
    segments[segments.length - 1].push({ x: point.x, y: point.y })
  })

  return <div className="forecast-mini-chart" aria-label="PM2.5 trajectory from now through six hours">
    <div className="mini-chart-title"><span>PM2.5 trajectory</span><small>Now → 1h → 3h → 6h</small></div>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img">
      <line className="chart-guide" x1={PAD_X} x2={WIDTH - PAD_X} y1={HEIGHT - PAD_Y} y2={HEIGHT - PAD_Y} />
      {segments.map((segment, index) => <polyline key={index} className="chart-line" points={segment.map(point => `${point.x},${point.y}`).join(' ')} />)}
      {coords.map(point => <g key={point.label} className={timeline === point.id ? 'selected' : ''}>
        {point.y != null ? <>
          <circle className="chart-point-halo" cx={point.x} cy={point.y} r={timeline === point.id ? 8 : 6} />
          <circle className="chart-point" cx={point.x} cy={point.y} r={timeline === point.id ? 4.5 : 3.5} />
          <text className="chart-value" x={point.x} y={Math.max(11, point.y - 11)} textAnchor="middle">{point.value?.toFixed(1)}</text>
        </> : <text className="chart-missing" x={point.x} y={HEIGHT / 2} textAnchor="middle">—</text>}
        <text className="chart-label" x={point.x} y={HEIGHT - 4} textAnchor="middle">{point.label}</text>
      </g>)}
    </svg>
  </div>
}
