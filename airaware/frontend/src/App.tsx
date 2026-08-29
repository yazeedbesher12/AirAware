import { useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertTriangle, BarChart3, CalendarRange, ChevronRight, CircleDot,
  BrainCircuit, CloudSun, Database, Download, FileText, FlaskConical, Gauge, HeartPulse, MapPin, Menu, RefreshCw,
  Route, Search, ShieldCheck, SlidersHorizontal, TimerOff, Waves, X,
} from 'lucide-react'
import Chart from './components/Chart'
import AdvancedAnalysis from './pages/AdvancedAnalysis'
import Phase3Page from './pages/Phase3'
import { api, downloadUrl, query, type Filters } from './services/api'

type AnyRow = Record<string, any>
type Overview = AnyRow & { sensor_timelines: AnyRow[]; locations: AnyRow[] }

const FEATURES = [
  ['pm25', 'PM2.5'], ['temperature', 'Temperature'], ['humidity', 'Humidity'],
  ['no2', 'NO₂'], ['o3', 'O₃'], ['aqi', 'Existing AQI'],
]
const COLORS = ['#0fa97b', '#f0a43b', '#4779d8', '#b85dc5', '#e05f5f', '#64756e']
const PAGE_META: Record<string, [string, string]> = {
  overview: ['Overview', 'Dataset coverage and system health at a glance'],
  quality: ['Data Quality', 'Missingness, structural availability, and evidence-based flags'],
  explorer: ['Sensor Explorer', 'Inspect raw measurements, rolling behavior, and distributions'],
  gaps: ['Time Gaps', 'Sampling continuity and exact sensor outage periods'],
  comparison: ['Sensor Comparison', 'Synchronized analysis across Nablus sensors'],
  aqi: ['AQI Investigation', 'Explore the source AQI without treating it as ground truth'],
  advanced: ['Advanced Analysis', 'Deep EDA, temporal dependency, multivariate structure, and anomaly consensus'],
  features: ['Feature Engineering', 'Evidence-driven lags, rolling context, quality features, and leakage controls'],
  who: ['WHO Health Analysis', 'Averaging-period health references with strict unit and coverage validation'],
  forecast: ['Forecast Dataset', 'Multi-horizon targets, event labels, and sample-validity preparation'],
  reports: ['Reports', 'Generated audit, cleaning, quality, and Phase 1 findings'],
}

const fmt = (value: unknown, digits = 1) => {
  if (value === null || value === undefined || value === '' || (typeof value === 'number' && !Number.isFinite(value))) return '—'
  return typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : String(value)
}
const dateFmt = (value: unknown) => value ? new Date(String(value)).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }) : '—'
const label = (feature: string) => FEATURES.find(([key]) => key === feature)?.[1] || feature

function StatusPill({ value }: { value: string }) {
  const kind = value.toLowerCase().replaceAll('_', '-')
  return <span className={`status status-${kind}`}>{value.replaceAll('_', ' ')}</span>
}

function Card({ title, eyebrow, action, children, className = '' }: { title?: string; eyebrow?: string; action?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return <section className={`card ${className}`}>
    {(title || eyebrow || action) && <div className="card-head"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}{title && <h3>{title}</h3>}</div>{action}</div>}
    {children}
  </section>
}

function Empty({ message }: { message: string }) {
  return <div className="empty"><CloudSun size={34} /><strong>No data to display</strong><span>{message}</span></div>
}

function Loading() {
  return <div className="loading"><span /><span /><span /><p>Loading local analysis…</p></div>
}

function Metric({ icon, label: text, value, sub, tone = 'green' }: { icon: React.ReactNode; label: string; value: string; sub: string; tone?: string }) {
  return <div className={`metric tone-${tone}`}><div className="metric-icon">{icon}</div><div><span>{text}</span><strong>{value}</strong><small>{sub}</small></div></div>
}

function GlobalFilters({ filters, setFilters, sensors }: { filters: Filters; setFilters: (next: Filters) => void; sensors: AnyRow[] }) {
  const visibleSensors = sensors.filter(sensor => !filters.city || sensor.city === filters.city)
  const update = (key: keyof Filters, value: string) => setFilters({ ...filters, [key]: value, ...(key === 'city' ? { sensorId: '' } : {}) })
  return <div className="filterbar">
    <div className="filter-title"><SlidersHorizontal size={16} /><span>Global filters</span></div>
    <label><span>City</span><select value={filters.city} onChange={e => update('city', e.target.value)}><option value="">All cities</option><option>Tulkarem</option><option>Nablus</option></select></label>
    <label><span>Sensor</span><select value={filters.sensorId} onChange={e => update('sensorId', e.target.value)}><option value="">All sensors</option>{visibleSensors.map(sensor => <option key={sensor.sensor_id} value={sensor.sensor_id}>#{sensor.sensor_id} · {sensor.location}</option>)}</select></label>
    <label><span>Feature</span><select value={filters.feature} onChange={e => update('feature', e.target.value)}>{FEATURES.map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select></label>
    <label><span>From</span><input type="date" value={filters.startTime} onChange={e => update('startTime', e.target.value)} /></label>
    <label><span>To</span><input type="date" value={filters.endTime} onChange={e => update('endTime', e.target.value)} /></label>
    <button className="filter-reset" onClick={() => setFilters({ city: '', sensorId: '', feature: 'pm25', startTime: '', endTime: '' })}><RefreshCw size={14} /> Reset</button>
  </div>
}

function OverviewPage({ overview, availability, sensors }: { overview: Overview; availability: AnyRow[]; sensors: AnyRow[] }) {
  const selectedIds = new Set(sensors.map(sensor => sensor.sensor_id))
  const timelines = overview.sensor_timelines.filter(row => selectedIds.has(row.sensor_id))
  const cityCounts = sensors.reduce((acc: Record<string, number>, row) => ({ ...acc, [row.city]: (acc[row.city] || 0) + row.readings }), {})
  return <div className="page-stack">
    <div className="metrics-grid">
      <Metric icon={<Database />} label="Total readings" value={fmt(overview.row_count, 0)} sub="source observations" />
      <Metric icon={<MapPin />} label="Coverage" value={`${overview.city_count} cities`} sub={`${overview.sensor_count} independent sensors`} tone="blue" />
      <Metric icon={<CalendarRange />} label="Observation span" value={`${fmt(overview.observation_duration_hours / 24, 1)} days`} sub={`${dateFmt(overview.earliest_timestamp)} UTC`} tone="amber" />
      <Metric icon={<Waves />} label="Expected interval" value={`${fmt(overview.expected_sampling_minutes, 0)} min`} sub="sensor-mode cadence" tone="violet" />
      <Metric icon={<CircleDot />} label="Environmental missing" value={`${fmt(overview.environmental_missing_percentage, 1)}%`} sub="includes structural absence" tone="blue" />
      <Metric icon={<TimerOff />} label="Major gaps" value={fmt(overview.major_gaps, 0)} sub="major or critical outages" tone="red" />
      <Metric icon={<AlertTriangle />} label="Review events" value={fmt(overview.suspicious_observations, 0)} sub="retained, not deleted" tone="amber" />
      <Metric icon={<ShieldCheck />} label="Source integrity" value="Verified" sub="SHA-256 preserved" />
    </div>
    <div className="grid-two">
      <Card eyebrow="Volume" title="Readings by city">
        <Chart data={[{ type: 'bar', x: Object.keys(cityCounts), y: Object.values(cityCounts), marker: { color: ['#0fa97b', '#4779d8'] }, hovertemplate: '%{x}<br>%{y:,} readings<extra></extra>' }]} layout={{ height: 300, yaxis: { title: { text: 'Readings' } } }} />
      </Card>
      <Card eyebrow="Sensor network" title="Readings by sensor">
        <Chart data={[{ type: 'bar', x: sensors.map(s => `#${s.sensor_id}`), y: sensors.map(s => s.readings), marker: { color: COLORS }, customdata: sensors.map(s => s.location), hovertemplate: '%{x} · %{customdata}<br>%{y:,} readings<extra></extra>' }]} layout={{ height: 300, yaxis: { title: { text: 'Readings' } } }} />
      </Card>
    </div>
    <Card eyebrow="Dataset timeline" title="Available range by sensor">
      <Chart data={timelines.map((sensor, i) => ({ type: 'scatter', mode: 'lines+markers', x: [sensor.start, sensor.end], y: [`#${sensor.sensor_id} · ${sensor.location}`, `#${sensor.sensor_id} · ${sensor.location}`], line: { color: COLORS[i], width: 8 }, marker: { size: 8 }, name: `Sensor ${sensor.sensor_id}`, hovertemplate: `%{x|%b %d, %H:%M} UTC<extra>Sensor ${sensor.sensor_id}</extra>` }))} layout={{ height: 310, showlegend: false, xaxis: { title: { text: 'UTC date' } } }} />
    </Card>
    <AvailabilityMatrix rows={availability} />
  </div>
}

function AvailabilityMatrix({ rows }: { rows: AnyRow[] }) {
  return <Card eyebrow="Sensor coverage matrix" title="Measurement availability">
    <div className="table-wrap"><table><thead><tr><th>Sensor</th><th>City / location</th>{FEATURES.map(([, text]) => <th key={text}>{text}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.sensor_id}><td><strong>#{row.sensor_id}</strong></td><td>{row.city}<small>{row.location}</small></td>{FEATURES.map(([feature]) => <td key={feature}><StatusPill value={row[feature]} /></td>)}</tr>)}</tbody>
    </table></div>
    <div className="legend-note"><span><i className="dot available" /> Available</span><span><i className="dot partial" /> Partially missing</span><span><i className="dot unavailable" /> Structurally unavailable</span><p>Structural absence reflects sensor/source capability. It is not treated as a random missing value and is never interpolated.</p></div>
  </Card>
}

function QualityPage({ missing, availability, sensors, events, suspicious }: { missing: AnyRow; availability: AnyRow[]; sensors: AnyRow[]; events: AnyRow; suspicious: AnyRow }) {
  const environmental = (missing?.per_column || []).filter((r: AnyRow) => FEATURES.some(([f]) => f === r.column))
  const perSensor = (missing?.per_sensor || []).filter((r: AnyRow) => r.classification !== 'STRUCTURALLY_UNAVAILABLE')
  const breakdown = [
    ['Suspicious / jump', events?.jumps?.length || 0], ['Possible stuck', events?.flatlines?.length || 0],
    ['Possible drift', events?.drifts?.length || 0], ['Gap', sensors.reduce((n, s) => n + s.gap_count, 0)],
    ['Normal rows', Math.max(0, 5636 - (suspicious?.total || 0))],
  ]
  const missingPeriods = (missing?.periods || []).filter((r: AnyRow) => r.classification !== 'STRUCTURALLY_UNAVAILABLE')
  return <div className="page-stack">
    <div className="grid-two">
      <Card eyebrow="Missing values" title="Missing percentage by feature">
        <Chart data={[{ type: 'bar', orientation: 'h', y: environmental.map((r: AnyRow) => label(r.column)), x: environmental.map((r: AnyRow) => r.missing_percentage), marker: { color: environmental.map((r: AnyRow) => r.missing_percentage > 50 ? '#d96161' : '#e3aa45') }, hovertemplate: '%{y}: %{x:.2f}%<extra></extra>' }]} layout={{ height: 330, xaxis: { title: { text: 'Missing %' } }, margin: { l: 95 } }} />
      </Card>
      <Card eyebrow="Quality flags" title="Evidence category breakdown">
        <Chart data={[{ type: 'bar', x: breakdown.map(x => x[0]), y: breakdown.map(x => x[1]), marker: { color: ['#e3aa45', '#b85dc5', '#4779d8', '#e05f5f', '#0fa97b'] }, hovertemplate: '%{x}<br>%{y:,}<extra></extra>' }]} layout={{ height: 330, yaxis: { title: { text: 'Events / rows' } } }} />
      </Card>
    </div>
    <Card eyebrow="Per-sensor completeness" title="Missingness by sensor and feature">
      <Chart data={[{ type: 'heatmap', x: FEATURES.map(([, t]) => t), y: sensors.map(s => `#${s.sensor_id} · ${s.location}`), z: sensors.map(s => FEATURES.map(([f]) => perSensor.find((r: AnyRow) => r.sensor_id === s.sensor_id && r.feature === f)?.missing_percentage ?? null)), colorscale: [[0, '#e9f7f1'], [.25, '#9ddcc7'], [.75, '#e8b14d'], [1, '#d96161']], colorbar: { title: { text: 'Missing %' } }, hovertemplate: '%{y}<br>%{x}: %{z:.2f}%<extra></extra>' }]} layout={{ height: 320, margin: { l: 175 } }} />
    </Card>
    <Card eyebrow="Missing data timeline" title="Observed missing values and sampling gaps">
      {missingPeriods.length ? <Chart data={missingPeriods.map((period: AnyRow) => ({ type: 'scatter', mode: 'lines+markers', x: [period.start_time, period.end_time], y: [`#${period.sensor_id} · ${label(period.feature)}`, `#${period.sensor_id} · ${label(period.feature)}`], line: { width: period.classification === 'TIME_GAP' ? 7 : 5, color: period.classification === 'TIME_GAP' ? '#d96161' : '#e3aa45' }, marker: { size: 5 }, name: period.classification, hovertemplate: `${period.classification}<br>%{x}<extra>Sensor ${period.sensor_id}</extra>`, showlegend: false }))} layout={{ height: 360, margin: { l: 120 }, xaxis: { title: { text: 'UTC timestamp' } } }} /> : <Empty message="No in-record missing values or cadence gaps were found for this selection." />}
    </Card>
    <AvailabilityMatrix rows={availability} />
    <Card eyebrow="Sensor quality" title="Transparent status summaries">
      <div className="sensor-cards">{sensors.map(sensor => <article className="sensor-card" key={sensor.sensor_id}><div className="sensor-card-top"><div><span>Sensor {sensor.sensor_id}</span><h4>{sensor.location}</h4><small>{sensor.city}</small></div><div className="status-stack">{sensor.statuses.map((status: string) => <StatusPill key={status} value={status} />)}</div></div><div className="sensor-facts"><span><b>{fmt(sensor.missing_rate, 2)}%</b> missing</span><span><b>{fmt(sensor.longest_gap_hours, 1)}h</b> longest gap</span><span><b>{sensor.sudden_jump_count}</b> jumps</span><span><b>{sensor.flatline_events}</b> stuck patterns</span></div><p>{sensor.explanation}</p></article>)}</div>
    </Card>
    <SuspiciousTable data={suspicious} />
  </div>
}

function SuspiciousTable({ data }: { data: AnyRow }) {
  return <Card eyebrow="Suspicious observation explorer" title={`${fmt(data?.total || 0, 0)} retained event records`} action={<a className="button small" href={downloadUrl('/api/download/suspicious')}><Download size={14} /> CSV</a>}>
    <div className="table-wrap dense"><table><thead><tr><th>Timestamp</th><th>City</th><th>Sensor</th><th>Feature</th><th>Value</th><th>Previous</th><th>Absolute Δ</th><th>Relative Δ</th><th>Flag</th><th>Reason</th></tr></thead><tbody>{(data?.items || []).slice(0, 100).map((row: AnyRow, i: number) => <tr key={`${row.timestamp}-${row.feature}-${i}`}><td>{dateFmt(row.timestamp)}</td><td>{row.city}</td><td>#{row.sensor_id}</td><td>{label(row.feature)}</td><td>{fmt(row.value, 2)}</td><td>{fmt(row.previous_value, 2)}</td><td>{fmt(row.absolute_change, 2)}</td><td>{row.relative_change == null ? '—' : `${fmt(row.relative_change * 100, 1)}%`}</td><td><StatusPill value={row.flag} /></td><td className="reason">{row.reason}</td></tr>)}</tbody></table></div>
  </Card>
}

function TimeSeriesChart({ data, feature, showRolling = true }: { data: AnyRow; feature: string; showRolling?: boolean }) {
  if (!data?.available) return <Empty message={data?.message || `${label(feature)} is not available for the selected sensors.`} />
  const traces: AnyRow[] = []
  data.series.forEach((series: AnyRow, i: number) => {
    traces.push({ type: 'scattergl', mode: 'lines', x: series.points.map((p: AnyRow) => p.timestamp), y: series.points.map((p: AnyRow) => p.value), name: `#${series.sensor_id} raw`, line: { color: COLORS[i], width: 1.3 }, customdata: series.points.map((p: AnyRow) => [series.sensor_id, p.overall_quality_flag]), hovertemplate: '%{x}<br>Sensor %{customdata[0]} · %{y:.2f}<br>%{customdata[1]}<extra></extra>', connectgaps: false })
    if (showRolling && series.points.some((p: AnyRow) => p.rolling_mean != null)) traces.push({ type: 'scattergl', mode: 'lines', x: series.points.map((p: AnyRow) => p.timestamp), y: series.points.map((p: AnyRow) => p.rolling_mean), name: `#${series.sensor_id} rolling`, line: { color: COLORS[i], width: 2.4, dash: 'dot' }, hoverinfo: 'skip' })
    const flagged = series.points.filter((p: AnyRow) => p.jump_flag || p.overall_quality_flag === 'SUSPICIOUS')
    if (flagged.length) traces.push({ type: 'scatter', mode: 'markers', x: flagged.map((p: AnyRow) => p.timestamp), y: flagged.map((p: AnyRow) => p.value), name: `#${series.sensor_id} flags`, marker: { color: '#db5f5f', size: 7, symbol: 'diamond-open' }, hovertemplate: '%{x}<br>%{y:.2f}<br>Flagged for review<extra></extra>' })
  })
  return <Chart data={traces} layout={{ height: 430, yaxis: { title: { text: `${label(feature)} (${data.unit})` } }, xaxis: { rangeslider: { visible: true, thickness: .08 }, title: { text: 'UTC timestamp' } } }} />
}

function ExplorerPage({ timeseries, distribution, filters, rolling, setRolling, events }: { timeseries: AnyRow; distribution: AnyRow; filters: Filters; rolling: string; setRolling: (v: string) => void; events: AnyRow }) {
  const sensorStats = distribution?.series || []
  const selectedJumps = (events?.jumps || []).filter((j: AnyRow) => j.feature === filters.feature && (!filters.city || j.city === filters.city) && (!filters.sensorId || String(j.sensor_id) === filters.sensorId)).slice(0, 20)
  const driftRows = (events?.drifts || []).filter((j: AnyRow) => j.feature === filters.feature && (!filters.city || j.city === filters.city) && (!filters.sensorId || String(j.sensor_id) === filters.sensorId))
  const rollingRows = (events?.rolling_summary || []).filter((j: AnyRow) => j.feature === filters.feature && (!filters.city || j.city === filters.city) && (!filters.sensorId || String(j.sensor_id) === filters.sensorId))
  return <div className="page-stack">
    <Card eyebrow="Raw + rolling behavior" title={`${label(filters.feature)} time series`} action={<label className="inline-control"><span>Rolling window</span><select value={rolling} onChange={e => setRolling(e.target.value)}><option value="none">None</option><option value="1h">1 hour</option><option value="3h">3 hours</option><option value="6h">6 hours</option><option value="12h">12 hours</option><option value="24h">24 hours</option></select></label>}>
      <TimeSeriesChart data={timeseries} feature={filters.feature} />
    </Card>
    {distribution?.available ? <>
      <div className="grid-two">
        <Card eyebrow="Distribution" title={`${label(filters.feature)} histogram`}><Chart data={sensorStats.map((s: AnyRow, i: number) => ({ type: 'histogram', x: s.sample, name: `#${s.sensor_id}`, opacity: .72, marker: { color: COLORS[i] }, nbinsx: 35 }))} layout={{ barmode: 'overlay', height: 330, xaxis: { title: { text: `${label(filters.feature)} (${distribution.unit})` } }, yaxis: { title: { text: 'Count' } } }} /></Card>
        <Card eyebrow="Distribution" title="Boxplot by sensor"><Chart data={sensorStats.map((s: AnyRow, i: number) => ({ type: 'box', y: s.sample, name: `#${s.sensor_id}`, marker: { color: COLORS[i] }, boxpoints: 'outliers', hovertemplate: '%{y:.2f}<extra>%{fullData.name}</extra>' }))} layout={{ height: 330, yaxis: { title: { text: `${label(filters.feature)} (${distribution.unit})` } } }} /></Card>
      </div>
      <Card eyebrow="Descriptive statistics" title="Selected sensor summary"><div className="stats-strip">{sensorStats.map((s: AnyRow) => <article key={s.sensor_id}><h4>Sensor {s.sensor_id}</h4><small>{s.location}</small>{Object.entries(s.statistics).map(([key, value]) => <span key={key}><em>{key.toUpperCase()}</em><b>{fmt(value, 2)}</b></span>)}</article>)}</div></Card>
    </> : <Card><Empty message={distribution?.message || `${label(filters.feature)} is unavailable for this selection.`} /></Card>}
    <div className="grid-two">
      <Card eyebrow="Sudden jump analysis" title="Strongest robust changes"><Chart data={[{ type: 'bar', orientation: 'h', y: selectedJumps.slice(0, 12).reverse().map((j: AnyRow) => `#${j.sensor_id} · ${dateFmt(j.timestamp)}`), x: selectedJumps.slice(0, 12).reverse().map((j: AnyRow) => j.absolute_change), marker: { color: '#e3aa45' }, customdata: selectedJumps.slice(0, 12).reverse().map((j: AnyRow) => j.robust_strength), hovertemplate: 'Δ %{x:.2f}<br>%{customdata:.1f}× robust threshold<extra></extra>' }]} layout={{ height: 360, margin: { l: 180 }, xaxis: { title: { text: `Absolute change (${timeseries?.unit || ''})` } } }} /></Card>
      <Card eyebrow="Rolling stability" title="Variability by time window"><Chart data={['1h', '3h', '6h', '12h', '24h'].map((window, i) => ({ type: 'bar', name: window, x: rollingRows.map((r: AnyRow) => `#${r.sensor_id}`), y: rollingRows.map((r: AnyRow) => r[`${window}_std_median`]), marker: { color: COLORS[i] }, hovertemplate: `${window} median rolling std<br>%{x}: %{y:.3f}<extra></extra>` }))} layout={{ height: 360, barmode: 'group', yaxis: { title: { text: 'Median rolling standard deviation' } } }} /></Card>
    </div>
    <Card eyebrow="Possible baseline change" title="Exploratory drift evidence">{driftRows.length ? <div className="event-list">{driftRows.map((row: AnyRow, i: number) => <article key={i}><Gauge /><div><strong>{row.label}</strong><span>Sensor #{row.sensor_id} · {label(row.feature)}</span><p>{dateFmt(row.start_time)} → {dateFmt(row.end_time)}</p><small>Median displacement: {fmt(row.magnitude, 2)} · {row.evidence}</small></div></article>)}</div> : <Empty message="No sustained baseline-shift period was flagged for this selection." />}</Card>
  </div>
}

function GapsPage({ gaps }: { gaps: AnyRow }) {
  const items = [...(gaps?.items || [])].sort((a, b) => b.duration_hours - a.duration_hours)
  const severityColors: Record<string, string> = { MINOR: '#e1b24e', MODERATE: '#e78349', MAJOR: '#d95f62', CRITICAL: '#872f47' }
  return <div className="page-stack">
    <div className="callout"><TimerOff /><div><strong>Gap classification is cadence-relative</strong><p>{gaps?.methodology}</p></div></div>
    <Card eyebrow="Availability timeline" title="Exact outage periods by sensor"><Chart data={items.map((gap: AnyRow) => ({ type: 'scatter', mode: 'lines', x: [gap.gap_start, gap.gap_end], y: [`#${gap.sensor_id} · ${gap.location}`, `#${gap.sensor_id} · ${gap.location}`], line: { color: severityColors[gap.severity], width: gap.severity === 'CRITICAL' ? 14 : 8 }, name: gap.severity, hovertemplate: `${gap.severity}<br>${fmt(gap.duration_hours, 2)} hours<br>%{x}<extra>Sensor ${gap.sensor_id}</extra>`, showlegend: false }))} layout={{ height: 360, xaxis: { title: { text: 'UTC timestamp' } }, margin: { l: 180 } }} /></Card>
    <div className="grid-two">
      <Card eyebrow="Gap distribution" title="Outage duration"><Chart data={[{ type: 'histogram', x: items.map(g => g.duration_hours), marker: { color: '#e26a54' }, nbinsx: 30, hovertemplate: '%{x:.2f} hours<br>%{y} gaps<extra></extra>' }]} layout={{ height: 310, xaxis: { title: { text: 'Duration (hours)' }, type: 'log' }, yaxis: { title: { text: 'Gap count' } } }} /></Card>
      <Card eyebrow="Sampling interval distribution" title="Observed cadence by sensor"><Chart data={(gaps?.sampling || []).map((s: AnyRow, i: number) => ({ type: 'bar', name: `#${s.sensor_id}`, x: (s.interval_distribution || []).map((d: AnyRow) => d.minutes), y: (s.interval_distribution || []).map((d: AnyRow) => d.count), marker: { color: COLORS[i] }, hovertemplate: `Sensor ${s.sensor_id}<br>%{x} minutes: %{y} intervals<extra></extra>` }))} layout={{ height: 310, barmode: 'group', xaxis: { title: { text: 'Interval (minutes)' }, type: 'log' }, yaxis: { title: { text: 'Count' }, type: 'log' } }} /></Card>
    </div>
    <Card eyebrow="Largest gaps" title={`${items.length} detected outages`} action={<a href={downloadUrl('/api/download/gaps')} className="button small"><Download size={14} /> CSV</a>}><div className="table-wrap"><table><thead><tr><th>Sensor</th><th>City / location</th><th>Gap start</th><th>Gap end</th><th>Duration</th><th>Missing expected</th><th>Severity</th></tr></thead><tbody>{items.map((gap: AnyRow, i: number) => <tr key={i} className={gap.severity === 'CRITICAL' ? 'critical-row' : ''}><td><strong>#{gap.sensor_id}</strong></td><td>{gap.city}<small>{gap.location}</small></td><td>{dateFmt(gap.gap_start)}</td><td>{dateFmt(gap.gap_end)}</td><td><strong>{fmt(gap.duration_hours, 2)} h</strong></td><td>{gap.missing_expected_readings}</td><td><StatusPill value={gap.severity} /></td></tr>)}</tbody></table></div></Card>
  </div>
}

function ComparisonPage({ comparison, timeseries, feature }: { comparison: AnyRow; timeseries: AnyRow; feature: string }) {
  const pairs = comparison?.pairs || []
  const sensors = [...new Set(pairs.flatMap((p: AnyRow) => [p.sensor_a, p.sensor_b]))].sort() as number[]
  const matrix = sensors.map(a => sensors.map(b => a === b ? 1 : (pairs.find((p: AnyRow) => (p.sensor_a === a && p.sensor_b === b) || (p.sensor_a === b && p.sensor_b === a))?.pearson ?? null)))
  return <div className="page-stack">
    <div className="callout info"><MapPin /><div><strong>Geographic context matters</strong><p>Nablus sensors are at different locations. Agreement and divergence are descriptive; location-driven differences are not labeled as sensor faults.</p></div></div>
    <Card eyebrow="Synchronized comparison" title={`Nablus ${label(feature)} time series`}><TimeSeriesChart data={timeseries} feature={feature} showRolling={false} /></Card>
    <div className="grid-two">
      <Card eyebrow="Cross-sensor agreement" title="Pearson correlation matrix"><Chart data={[{ type: 'heatmap', x: sensors.map(s => `Sensor ${s}`), y: sensors.map(s => `Sensor ${s}`), z: matrix, zmin: -1, zmax: 1, colorscale: [[0, '#d95f62'], [.5, '#f1f3f2'], [1, '#0fa97b']], text: matrix.map(row => row.map(v => v == null ? '—' : v.toFixed(3))), texttemplate: '%{text}', hovertemplate: '%{y} ↔ %{x}<br>Pearson %{z:.3f}<extra></extra>' }]} layout={{ height: 360 }} /></Card>
      <Card eyebrow="Pairwise divergence" title="Mean absolute difference"><Chart data={[{ type: 'bar', x: pairs.map((p: AnyRow) => p.sensor_pair), y: pairs.map((p: AnyRow) => p.mean_absolute_difference), marker: { color: COLORS }, customdata: pairs.map((p: AnyRow) => p.overlap_count), hovertemplate: '%{x}<br>MAE %{y:.2f}<br>n=%{customdata:,}<extra></extra>' }]} layout={{ height: 360, yaxis: { title: { text: 'Mean absolute difference' } } }} /></Card>
    </div>
    <Card eyebrow="Pairwise statistics" title={`${label(feature)} overlap and differences`}><div className="table-wrap"><table><thead><tr><th>Sensor pair</th><th>Overlap</th><th>Pearson</th><th>Spearman</th><th>MAE</th><th>Median abs. diff.</th><th>Relative diff.</th></tr></thead><tbody>{pairs.map((p: AnyRow) => <tr key={p.sensor_pair}><td><strong>{p.sensor_pair}</strong></td><td>{fmt(p.overlap_count, 0)}</td><td>{fmt(p.pearson, 3)}</td><td>{fmt(p.spearman, 3)}</td><td>{fmt(p.mean_absolute_difference, 2)}</td><td>{fmt(p.median_absolute_difference, 2)}</td><td>{p.mean_relative_difference == null ? '—' : `${fmt(p.mean_relative_difference * 100, 1)}%`}</td></tr>)}</tbody></table></div></Card>
  </div>
}

function AqiPage({ analysis, timeseries, distribution, scatter, pollutantScatters }: { analysis: AnyRow; timeseries: AnyRow; distribution: AnyRow; scatter: AnyRow; pollutantScatters: Record<string, AnyRow> }) {
  const stats = analysis?.statistics || {}
  const allSamples = (distribution?.series || []).flatMap((s: AnyRow) => s.sample)
  return <div className="page-stack">
    <div className="aqi-warning"><AlertTriangle /><div><strong>AQI methodology from source dataset has not been verified.</strong><p>This page investigates the existing column only. It is not recalculated, trusted as ground truth, or interpreted against health thresholds.</p></div></div>
    <div className="metrics-grid compact">
      <Metric icon={<Gauge />} label="AQI range" value={`${fmt(stats.min, 0)}–${fmt(stats.max, 0)}`} sub="source values" tone="amber" />
      <Metric icon={<BarChart3 />} label="Median AQI" value={fmt(stats.median, 0)} sub={`mean ${fmt(stats.mean, 1)}`} tone="blue" />
      <Metric icon={<AlertTriangle />} label="Above 500" value={fmt(stats.above_500_count, 0)} sub="requires source review" tone="red" />
    </div>
    <Card eyebrow="Existing AQI" title="Time-series behavior"><TimeSeriesChart data={timeseries} feature="aqi" showRolling={false} /></Card>
    <div className="grid-two">
      <Card eyebrow="Distribution" title="AQI histogram"><Chart data={[{ type: 'histogram', x: allSamples, nbinsx: 50, marker: { color: '#e2a541' }, hovertemplate: 'AQI %{x}<br>%{y} readings<extra></extra>' }]} layout={{ height: 330, xaxis: { title: { text: 'Existing AQI' } }, yaxis: { title: { text: 'Count' } } }} /></Card>
      <Card eyebrow="By sensor" title="AQI boxplot"><Chart data={(distribution?.series || []).map((s: AnyRow, i: number) => ({ type: 'box', y: s.sample, name: `#${s.sensor_id}`, marker: { color: COLORS[i] }, boxpoints: 'outliers' }))} layout={{ height: 330, yaxis: { title: { text: 'Existing AQI' } } }} /></Card>
    </div>
    <Card eyebrow="Relationship investigation" title="AQI vs PM2.5"><ScatterChart data={scatter} xLabel="PM2.5" yLabel="Existing AQI" /></Card>
    <div className="grid-two"><Card eyebrow="Tulkarem relationship" title="AQI vs NO₂"><ScatterChart data={pollutantScatters.no2} xLabel="NO₂" yLabel="Existing AQI" /></Card><Card eyebrow="Tulkarem relationship" title="AQI vs O₃"><ScatterChart data={pollutantScatters.o3} xLabel="O₃" yLabel="Existing AQI" /></Card></div>
    <Card eyebrow="Extreme source values" title="Top AQI observations"><div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>City</th><th>Sensor</th><th>AQI</th><th>PM2.5</th><th>Temperature</th><th>Humidity</th><th>NO₂</th><th>O₃</th></tr></thead><tbody>{(analysis?.top_observations || []).slice(0, 25).map((r: AnyRow, i: number) => <tr key={i} className={r.aqi > 500 ? 'critical-row' : ''}><td>{dateFmt(r.timestamp)}</td><td>{r.city_name}</td><td>#{r.sensor_id}</td><td><strong>{r.aqi}</strong></td><td>{fmt(r.pm25, 2)}</td><td>{fmt(r.temperature, 2)}</td><td>{fmt(r.humidity, 2)}</td><td>{fmt(r.no2, 2)}</td><td>{fmt(r.o3, 2)}</td></tr>)}</tbody></table></div></Card>
  </div>
}

function ScatterChart({ data, xLabel, yLabel }: { data: AnyRow; xLabel: string; yLabel: string }) {
  if (!data?.available) return <Empty message={`${xLabel} is not available for the selected sensors.`} />
  return <Chart data={data.series.map((s: AnyRow, i: number) => ({ type: 'scattergl', mode: 'markers', x: s.points.map((p: AnyRow) => p.x), y: s.points.map((p: AnyRow) => p.y), name: `#${s.sensor_id}`, marker: { color: COLORS[i], size: 5, opacity: .52 }, hovertemplate: `${xLabel} %{x:.2f}<br>${yLabel} %{y:.2f}<extra>Sensor ${s.sensor_id}</extra>` }))} layout={{ height: 380, xaxis: { title: { text: xLabel } }, yaxis: { title: { text: yLabel } }, hovermode: 'closest' }} />
}

function ReportsPage({ reports, overview }: { reports: AnyRow[]; overview: Overview }) {
  const descriptions: Record<string, string> = {
    '01_data_audit_report.md': 'Raw structure, coverage, timestamp integrity, memory, and duplicate interpretation.',
    '02_cleaning_report.md': 'Conservative cleaning decisions and complete accounting of changes.',
    '03_sensor_quality_report.md': 'Per-sensor availability, gaps, pattern flags, and descriptive status.',
    '04_phase1_findings.md': 'Evidence-led findings and concerns to resolve before Advanced EDA.',
  }
  return <div className="page-stack">
    <div className="report-hero"><div><span className="eyebrow">Local evidence pack</span><h2>Phase 1 reports are ready</h2><p>Each report is generated directly from the pipeline results. The original CSV checksum remains <code>{overview.source_sha256?.slice(0, 16)}…</code>.</p></div><a className="button primary" href={downloadUrl('/api/download/cleaned')}><Download size={17} /> Download cleaned CSV</a></div>
    <div className="reports-grid">{reports.map((report, i) => <a className={`report-card ${i === 3 ? 'featured' : ''}`} key={report.name} href={downloadUrl(report.download_url)}><div className="report-icon"><FileText /></div><div><span>{i === 3 ? 'Priority report' : `Report 0${i + 1}`}</span><h3>{report.name.replace(/\d+_|\.md/g, '').replaceAll('_', ' ')}</h3><p>{descriptions[report.name]}</p></div><Download size={18} /></a>)}</div>
    <Card eyebrow="Cleaning ledger" title="Conservative cleaning result"><div className="ledger"><span><em>Original rows</em><b>{fmt(overview.row_count, 0)}</b></span><span><em>Cleaned rows</em><b>{fmt(overview.row_count, 0)}</b></span><span><em>Rows removed</em><b>0</b></span><span><em>Values modified</em><b>0</b></span><span><em>Suspicious retained</em><b>{fmt(overview.suspicious_observations, 0)}</b></span></div></Card>
  </div>
}

export default function App() {
  const [page, setPage] = useState('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [overview, setOverview] = useState<Overview | null>(null)
  const [sensors, setSensors] = useState<AnyRow[]>([])
  const [availability, setAvailability] = useState<AnyRow[]>([])
  const [missing, setMissing] = useState<AnyRow>({})
  const [gaps, setGaps] = useState<AnyRow>({})
  const [events, setEvents] = useState<AnyRow>({})
  const [suspicious, setSuspicious] = useState<AnyRow>({ items: [] })
  const [reports, setReports] = useState<AnyRow[]>([])
  const [aqiAnalysis, setAqiAnalysis] = useState<AnyRow>({})
  const [timeseries, setTimeseries] = useState<AnyRow>({ series: [] })
  const [distribution, setDistribution] = useState<AnyRow>({ series: [] })
  const [comparison, setComparison] = useState<AnyRow>({ pairs: [] })
  const [scatter, setScatter] = useState<AnyRow>({ series: [] })
  const [pollutantScatters, setPollutantScatters] = useState<Record<string, AnyRow>>({ no2: {}, o3: {} })
  const [rolling, setRolling] = useState('3h')
  const [filters, setFilters] = useState<Filters>({ city: '', sensorId: '', feature: 'pm25', startTime: '', endTime: '' })

  useEffect(() => {
    Promise.all([
      api<Overview>('/api/overview'), api<AnyRow[]>('/api/sensors'), api<AnyRow[]>('/api/availability'),
      api<AnyRow>('/api/missing'), api<AnyRow>('/api/gaps'), api<AnyRow>('/api/analysis-events'),
      api<AnyRow>('/api/suspicious?limit=200'), api<AnyRow[]>('/api/reports'), api<AnyRow>('/api/aqi-analysis'),
    ]).then(([o, s, a, m, g, e, x, r, q]) => { setOverview(o); setSensors(s); setAvailability(a); setMissing(m); setGaps(g); setEvents(e); setSuspicious(x); setReports(r); setAqiAnalysis(q) }).catch(err => setError(err.message)).finally(() => setLoading(false))
  }, [])

  const params = useMemo(() => ({ city: filters.city, sensor_id: filters.sensorId, start_time: filters.startTime, end_time: filters.endTime ? `${filters.endTime}T23:59:59Z` : '' }), [filters.city, filters.sensorId, filters.startTime, filters.endTime])
  useEffect(() => {
    if (loading) return
    const cityForComparison = page === 'comparison' ? 'Nablus' : filters.city
    const featureForPage = page === 'aqi' ? 'aqi' : page === 'comparison' ? (['pm25', 'temperature', 'humidity', 'aqi'].includes(filters.feature) ? filters.feature : 'pm25') : filters.feature
    Promise.all([
      api<AnyRow>(`/api/timeseries${query({ ...params, city: cityForComparison, feature: featureForPage, rolling_window: page === 'explorer' ? rolling : 'none' })}`),
      api<AnyRow>(`/api/distribution${query({ ...params, feature: featureForPage })}`),
      api<AnyRow>(`/api/nablus-comparison${query({ feature: ['pm25', 'temperature', 'humidity', 'aqi'].includes(filters.feature) ? filters.feature : 'pm25' })}`),
    ]).then(([t, d, c]) => { setTimeseries(t); setDistribution(d); setComparison(c) }).catch(err => setError(err.message))
  }, [filters, loading, page, rolling])

  useEffect(() => {
    if (loading || page !== 'quality') return
    api<AnyRow>(`/api/suspicious${query({ ...params, feature: filters.feature, limit: 200 })}`).then(setSuspicious).catch(err => setError(err.message))
  }, [loading, page, params, filters.feature])

  useEffect(() => {
    if (loading || page !== 'aqi') return
    Promise.all([
      api<AnyRow>(`/api/scatter${query({ ...params, x_feature: 'pm25', y_feature: 'aqi' })}`),
      api<AnyRow>(`/api/scatter${query({ city: filters.city || 'Tulkarem', sensor_id: filters.sensorId, x_feature: 'no2', y_feature: 'aqi' })}`),
      api<AnyRow>(`/api/scatter${query({ city: filters.city || 'Tulkarem', sensor_id: filters.sensorId, x_feature: 'o3', y_feature: 'aqi' })}`),
    ]).then(([pm, no2, o3]) => { setScatter(pm); setPollutantScatters({ no2, o3 }) }).catch(err => setError(err.message))
  }, [loading, page, params, filters.city, filters.sensorId])

  if (loading) return <div className="boot"><div className="brand-mark"><Waves /></div><Loading /></div>
  if (error || !overview) return <div className="boot"><div className="error-box"><AlertTriangle /><h2>Local backend unavailable</h2><p>{error || 'No overview data was returned.'}</p><code>uvicorn app.main:app --reload</code><button onClick={() => location.reload()}>Try again</button></div></div>

  const nav = [
    ['overview', Activity, 'Overview'], ['quality', ShieldCheck, 'Data Quality'], ['explorer', Search, 'Sensor Explorer'],
    ['gaps', TimerOff, 'Time Gaps'], ['comparison', Waves, 'Sensor Comparison'], ['aqi', Gauge, 'AQI Investigation'], ['advanced', BrainCircuit, 'Advanced Analysis'],
    ['features', FlaskConical, 'Feature Engineering'], ['who', HeartPulse, 'WHO Health Analysis'], ['forecast', Route, 'Forecast Dataset'], ['reports', FileText, 'Reports'],
  ] as const
  const comparisonFeature = ['pm25', 'temperature', 'humidity', 'aqi'].includes(filters.feature) ? filters.feature : 'pm25'
  const visibleAvailability = availability.filter(row => (!filters.city || row.city === filters.city) && (!filters.sensorId || String(row.sensor_id) === filters.sensorId))

  return <div className="app-shell">
    <aside className={menuOpen ? 'sidebar open' : 'sidebar'}>
      <div className="brand"><div className="brand-mark"><Waves /></div><div><strong>AirAware</strong><span>Sensor intelligence</span></div><button className="close-menu" onClick={() => setMenuOpen(false)}><X /></button></div>
      <div className="local-badge"><CircleDot /> Local workspace<span>Phase 3</span></div>
      <nav>{nav.map(([key, Icon, text]) => <button key={key} className={page === key ? 'active' : ''} onClick={() => { setPage(key); setMenuOpen(false) }}><Icon /><span>{text}</span>{page === key && <ChevronRight />}</button>)}</nav>
      <div className="sidebar-foot"><div><span>Data window</span><strong>{fmt(overview.observation_duration_hours / 24, 1)} days</strong></div><div><span>Last reading</span><strong>{new Date(overview.latest_timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })}</strong></div><small>All processing stays on this machine.</small></div>
    </aside>
    {menuOpen && <button className="scrim" onClick={() => setMenuOpen(false)} aria-label="Close navigation" />}
    <main>
      <header><button className="menu-button" onClick={() => setMenuOpen(true)}><Menu /></button><div><span className="breadcrumb">AirAware / Local Phases 1-3</span><h1>{PAGE_META[page][0]}</h1><p>{PAGE_META[page][1]}</p></div><div className="header-status"><span><i /> API connected</span><small>Local · UTC data</small></div></header>
      <GlobalFilters filters={filters} setFilters={setFilters} sensors={sensors} />
      <div className="content">
        {page === 'overview' && <OverviewPage overview={overview} availability={visibleAvailability} sensors={sensors.filter(s => (!filters.city || s.city === filters.city) && (!filters.sensorId || String(s.sensor_id) === filters.sensorId))} />}
        {page === 'quality' && <QualityPage missing={missing} availability={visibleAvailability} sensors={sensors.filter(s => (!filters.city || s.city === filters.city) && (!filters.sensorId || String(s.sensor_id) === filters.sensorId))} events={events} suspicious={suspicious} />}
        {page === 'explorer' && <ExplorerPage timeseries={timeseries} distribution={distribution} filters={filters} rolling={rolling} setRolling={setRolling} events={events} />}
        {page === 'gaps' && <GapsPage gaps={{ ...gaps, items: gaps.items.filter((g: AnyRow) => (!filters.city || g.city === filters.city) && (!filters.sensorId || String(g.sensor_id) === filters.sensorId)) }} />}
        {page === 'comparison' && <ComparisonPage comparison={comparison} timeseries={timeseries} feature={comparisonFeature} />}
        {page === 'aqi' && <AqiPage analysis={aqiAnalysis} timeseries={timeseries} distribution={distribution} scatter={scatter} pollutantScatters={pollutantScatters} />}
        {page === 'advanced' && <AdvancedAnalysis filters={filters} />}
        {page === 'features' && <Phase3Page view="features" filters={filters} />}
        {page === 'who' && <Phase3Page view="who" filters={filters} />}
        {page === 'forecast' && <Phase3Page view="forecast" filters={filters} />}
        {page === 'reports' && <ReportsPage reports={reports} overview={overview} />}
      </div>
    </main>
  </div>
}
