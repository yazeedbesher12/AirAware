import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CheckCircle2, Clock3, Download, FlaskConical, HeartPulse, Route, ShieldCheck } from 'lucide-react'
import Chart from '../components/Chart'
import { api, downloadUrl, query, type Filters } from '../services/api'

type Row = Record<string, any>
export type Phase3View = 'features' | 'who' | 'forecast'

const COLORS = ['#0fa97b', '#4779d8', '#f0a43b', '#b85dc5', '#e05f5f']
const fmt = (value: unknown, digits = 2) => value == null || value === '' || (typeof value === 'number' && !Number.isFinite(value)) ? '-' : typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : String(value)
const dateText = (value: unknown) => value ? new Date(String(value)).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }) : '-'
const groupBySensor = (items: Row[]): Row[][] => Object.values(items.reduce((all: Record<string, Row[]>, row: Row) => { (all[String(row.sensor_id)] ||= []).push(row); return all }, {}))

function Panel({ eyebrow, title, action, children }: { eyebrow: string; title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return <section className="card"><div className="card-head"><div><span className="eyebrow">{eyebrow}</span><h3>{title}</h3></div>{action}</div>{children}</section>
}

function Kpis({ items }: { items: { label: string; value: string; note: string; tone?: string }[] }) {
  return <div className="phase3-kpis">{items.map(item => <article className={item.tone || ''} key={item.label}><small>{item.label}</small><strong>{item.value}</strong><span>{item.note}</span></article>)}</div>
}

function DataTable({ rows, columns, limit = 60 }: { rows: Row[]; columns: [string, string][]; limit?: number }) {
  return <div className="table-wrap dense"><table><thead><tr>{columns.map(([, label]) => <th key={label}>{label}</th>)}</tr></thead><tbody>{rows.slice(0, limit).map((row, index) => <tr key={index}>{columns.map(([key]) => <td key={key}>{key.includes('timestamp') ? dateText(row[key]) : typeof row[key] === 'number' ? fmt(row[key], 3) : fmt(row[key])}</td>)}</tr>)}</tbody></table></div>
}

function Notice({ title, text, warning = false }: { title: string; text: string; warning?: boolean }) {
  return <aside className={`phase3-notice ${warning ? 'warning' : ''}`}>{warning ? <AlertTriangle /> : <ShieldCheck />}<div><strong>{title}</strong><p>{text}</p></div></aside>
}

function Loading() { return <div className="advanced-loading"><span /><p>Loading local Phase 3 outputs...</p></div> }
function Empty({ text }: { text: string }) { return <div className="advanced-empty"><AlertTriangle /><strong>No valid records for this selection</strong><span>{text}</span></div> }

function FeatureEngineering({ filters }: { filters: Filters }) {
  const [summary, setSummary] = useState<Row | null>(null)
  const [catalog, setCatalog] = useState<Row>({ items: [] })
  const [ranking, setRanking] = useState<Row>({ items: [] })
  const [redundancy, setRedundancy] = useState<Row>({ items: [] })
  const [target, setTarget] = useState('target_pm25_1h')
  const [error, setError] = useState('')
  useEffect(() => {
    Promise.all([
      api<Row>('/api/ml/summary'), api<Row>('/api/ml/features'),
      api<Row>(`/api/ml/feature-ranking${query({ target, limit: 100 })}`), api<Row>('/api/ml/redundancy'),
    ]).then(([s, c, r, d]) => { setSummary(s); setCatalog(c); setRanking(r); setRedundancy(d) }).catch(err => setError(err.message))
  }, [target])
  if (error) return <Notice warning title="Feature outputs unavailable" text={error} />
  if (!summary) return <Loading />
  const families = summary.feature_engineering.families || {}
  const ranked = ranking.items || []
  return <div className="phase3-stack">
    <Kpis items={[
      { label: 'Candidate features', value: fmt(summary.feature_engineering.total_candidate_features, 0), note: 'evidence-driven, not exhaustive' },
      { label: 'Lag features', value: fmt(families.LAG || 0, 0), note: 'exact elapsed-time joins' },
      { label: 'Rolling features', value: fmt(families.ROLLING || 0, 0), note: 'backward-looking only' },
      { label: 'Redundant pairs', value: fmt(summary.feature_engineering.redundant_pairs, 0), note: 'flagged, not automatically removed', tone: 'amber' },
    ]} />
    <Notice title="What Feature Engineering means" text="It converts cleaned observations into forecasting inputs that are known at prediction time—for example PM2.5 one hour ago, the backward 3-hour mean, recent variability, and sensor-quality context. The original measurements remain unchanged." />
    <Notice title="Leakage control passed" text="Targets are stored separately. Exact-time lags and backward rolling windows reset at continuous segment boundaries; no centered or future-derived feature is present." />
    <div className="grid-two feature-overview-grid">
      <Panel eyebrow="Phase 2 evidence" title="Selected lag design"><div className="feature-plan">{Object.entries(summary.feature_engineering.selected_lags || {}).map(([feature, lags]) => <article key={feature}><strong>{feature.toUpperCase()}</strong><span>{(lags as number[]).map(value => value < 60 ? `${value}m` : `${value / 60}h`).join(' / ')}</span><p>{feature === 'pm25' ? 'Strong persistence was measured through 1-6 hours.' : feature === 'no2' || feature === 'o3' ? 'Created only for Tulkarem; Nablus remains structurally unavailable.' : 'Phase 2 showed strong short-term persistence and cross-feature dependency.'}</p></article>)}</div></Panel>
      <Panel eyebrow="Candidate ranking" title="Association with future PM2.5" action={<label className="inline-control"><span>Target</span><select value={target} onChange={e => setTarget(e.target.value)}><option>target_pm25_1h</option><option>target_pm25_3h</option><option>target_pm25_6h</option></select></label>}>
        <Chart data={[{ type: 'bar', orientation: 'h', y: ranked.slice(0, 12).reverse().map((row: Row) => row.feature), x: ranked.slice(0, 12).reverse().map((row: Row) => row.score), marker: { color: ranked.slice(0, 12).reverse().map((row: Row) => row.association_method === 'mutual_information' ? '#4779d8' : '#0fa97b') }, customdata: ranked.slice(0, 12).reverse().map((row: Row) => row.association_method), hovertemplate: '%{y}<br>%{customdata}: %{x:.3f}<extra></extra>' }]} layout={{ height: 390, margin: { l: 190 }, xaxis: { title: { text: 'Association score' } } }} />
      </Panel>
    </div>
    <Panel eyebrow="Feature catalog" title="Why each feature was created" action={<a className="button small" href={downloadUrl('/api/ml/download/feature_catalog.csv')}><Download size={14} /> Catalog CSV</a>}><DataTable rows={catalog.items || []} columns={[["feature", "Feature"], ["family", "Family"], ["availability_scope", "Availability"], ["reason", "Why created"], ["evidence", "Evidence"], ["leakage_check", "Leakage"]]} /></Panel>
    <Panel eyebrow="Redundancy review" title="Highly correlated engineered pairs"><DataTable rows={redundancy.items || []} columns={[["feature_a", "Feature A"], ["feature_b", "Feature B"], ["correlation", "Correlation"], ["recommendation", "Recommendation"]]} /></Panel>
    <Notice warning title="Selection is not final" text="The displayed ranking uses the complete Phase 3 dataset for exploratory screening. Phase 4 must repeat selection, imputation, and scaling inside chronological training folds only." />
  </div>
}

function WhoHealth({ filters }: { filters: Filters }) {
  const [pollutant, setPollutant] = useState('pm25')
  const [status, setStatus] = useState<Row | null>(null)
  const [series, setSeries] = useState<Row | null>(null)
  const [coverage, setCoverage] = useState<Row>({ items: [] })
  const [exceedance, setExceedance] = useState<Row>({})
  const [error, setError] = useState('')
  const params = useMemo(() => ({ city: filters.city, sensor_id: filters.sensorId, start_time: filters.startTime, end_time: filters.endTime ? `${filters.endTime}T23:59:59Z` : '' }), [filters])
  useEffect(() => {
    Promise.all([
      api<Row>(`/api/who/status${query({ city: filters.city, sensor_id: filters.sensorId })}`),
      api<Row>(`/api/who/timeseries${query({ ...params, pollutant, max_points: 2000 })}`),
      api<Row>(`/api/who/coverage${query({ city: filters.city, sensor_id: filters.sensorId, pollutant })}`),
      api<Row>(`/api/who/exceedances${query({ ...params, pollutant, limit: 100 })}`),
    ]).then(([s, t, c, e]) => { setStatus(s); setSeries(t); setCoverage(c); setExceedance(e) }).catch(err => setError(err.message))
  }, [filters.city, filters.sensorId, pollutant, params])
  if (error) return <Notice warning title="WHO analysis unavailable" text={error} />
  if (!status || !series) return <Loading />
  const rows = series.items || []
  const config = (series.config || [])[0]
  const groups = groupBySensor(rows)
  const compatible = String(config?.unit_compatible).toLowerCase() === 'true'
  return <div className="phase3-stack">
    <Notice warning title="WHO status uses averaging periods" text="Single instantaneous readings are never compared with the WHO 24-hour or 8-hour guideline. Results are health-reference screening, not an AQI or legal compliance determination." />
    <div className="who-toolbar"><label className="inline-control"><span>WHO pollutant analysis</span><select value={pollutant} onChange={e => setPollutant(e.target.value)}><option value="pm25">PM2.5 - 24h</option><option value="no2">NO2 - 24h (unit unverified)</option><option value="o3">O3 - 8h (unit unverified)</option></select></label><a className="button small" href={downloadUrl('/api/ml/download/who_analysis_results.csv')}><Download size={14} /> WHO results</a></div>
    <Kpis items={[
      { label: 'Valid windows', value: fmt(exceedance.valid_windows || 0, 0), note: `${pollutant.toUpperCase()} reference windows` },
      { label: 'Above guideline', value: compatible ? fmt(exceedance.exceedance_windows || 0, 0) : 'Not compared', note: compatible ? `${fmt(exceedance.exceedance_percentage || 0, 1)}% of valid windows` : 'source unit is unverified', tone: compatible ? 'amber' : 'red' },
      { label: 'Guideline', value: config ? `${fmt(config.guideline)} ${config.unit}` : '-', note: config?.averaging_period || 'not configured' },
      { label: 'Minimum coverage', value: config ? `${fmt(config.minimum_window_coverage_percent, 0)}%` : '-', note: 'within continuous segment' },
    ]} />
    {!compatible && <Notice warning title="WHO comparison not available in current unit" text="The CSV and project source do not document whether this gas is stored in ug/m3 or ppb. No conversion, ratio, exceedance, or guideline line is generated." />}
    <Panel eyebrow="AirAware WHO-based Health Status" title="Latest valid interpretation by sensor"><div className="who-status-grid">{(status.items || []).filter((row: Row) => row.pollutant === pollutant).map((row: Row) => <article key={`${row.sensor_id}-${row.pollutant}`}><div><HeartPulse /><span>Sensor {row.sensor_id}</span></div><strong>{row.status}</strong><p>{row.average == null ? 'No valid average' : `${fmt(row.average)} ${row.dataset_unit}`} <small>{fmt(row.coverage_percent, 1)}% coverage</small></p><em>{row.unit_compatible ? `Ratio ${fmt(row.ratio)}x guideline` : 'Unit comparison disabled'}</em></article>)}</div></Panel>
    <Panel eyebrow="Averaging-period comparison" title={`${pollutant.toUpperCase()} rolling average${compatible ? ' vs WHO guideline' : ''}`}>
      {rows.length ? <Chart data={[
        ...groups.map((group: Row[], index) => ({ type: 'scattergl', mode: 'lines', x: group.map(row => row.timestamp), y: group.map(row => row.average), name: `Sensor ${group[0].sensor_id}`, line: { color: COLORS[index], width: 1.8 }, connectgaps: false, customdata: group.map(row => [row.coverage_percent, row.status]), hovertemplate: '%{x}<br>%{y:.2f}<br>Coverage %{customdata[0]:.1f}%<br>%{customdata[1]}<extra></extra>' })),
        ...(compatible && config ? [{ type: 'scatter', mode: 'lines', x: rows.map((row: Row) => row.timestamp), y: rows.map(() => config.guideline), name: 'WHO AQG', line: { color: '#d95d5d', dash: 'dash', width: 2 } }] : []),
      ]} layout={{ height: 430, yaxis: { title: { text: `${config?.averaging_period || ''} average (${config?.dataset_unit || ''})` } }, xaxis: { rangeslider: { visible: true, thickness: .07 } } }} /> : <Empty text="No averaging-window data exists for the selected sensors." />}
    </Panel>
    <Panel eyebrow="Coverage quality" title="Valid averaging windows per sensor"><DataTable rows={coverage.items || []} columns={[["sensor_id", "Sensor"], ["city", "City"], ["total_windows", "Total windows"], ["valid_windows", "Valid windows"], ["average_coverage_percent", "Average coverage %"]]} /></Panel>
  </div>
}

function ForecastDataset({ filters }: { filters: Filters }) {
  const [horizon, setHorizon] = useState('1h')
  const [validity, setValidity] = useState<Row | null>(null)
  const [targets, setTargets] = useState<Row | null>(null)
  const [error, setError] = useState('')
  const params = useMemo(() => ({ city: filters.city, sensor_id: filters.sensorId, start_time: filters.startTime, end_time: filters.endTime ? `${filters.endTime}T23:59:59Z` : '' }), [filters])
  useEffect(() => {
    Promise.all([
      api<Row>(`/api/ml/sample-validity${query({ city: filters.city, sensor_id: filters.sensorId, horizon })}`),
      api<Row>(`/api/ml/targets${query({ ...params, horizon, max_points: 1800 })}`),
    ]).then(([v, t]) => { setValidity(v); setTargets(t) }).catch(err => setError(err.message))
  }, [filters.city, filters.sensorId, horizon, params])
  if (error) return <Notice warning title="Forecast dataset unavailable" text={error} />
  if (!validity || !targets) return <Loading />
  const total = (validity.totals || [])[0] || {}
  const points = (targets.items || []).filter((row: Row) => row[`target_pm25_${horizon}`] != null)
  const bySensor = groupBySensor(points)
  return <div className="phase3-stack">
    <div className="forecast-toolbar"><div><span className="eyebrow">Multi-horizon design</span><h2>Forecast sample preparation</h2><p>Features at t use only current/past data. Concentrations and event labels live in the separate future-target file.</p></div><label className="inline-control"><span>Forecast horizon</span><select value={horizon} onChange={e => setHorizon(e.target.value)}><option value="1h">1 hour</option><option value="3h">3 hours</option><option value="6h">6 hours</option></select></label></div>
    <Kpis items={[
      { label: 'Potential timestamps', value: fmt(total.total_timestamps || 0, 0), note: 'before validity controls' },
      { label: `Valid ${horizon} samples`, value: fmt(total.valid_training_samples || 0, 0), note: `${total.total_timestamps ? fmt(total.valid_training_samples / total.total_timestamps * 100, 1) : 0}% retained` },
      { label: 'Valid event labels', value: fmt(total.valid_event_labels || 0, 0), note: 'future valid 24h WHO window' },
      { label: 'Positive event labels', value: fmt(total.positive_event_labels || 0, 0), note: 'future rolling PM2.5 above AQG', tone: 'amber' },
    ]} />
    <div className="forecast-flow"><div><span>LOOKBACK</span><strong>6h history</strong><small>lags + rolling + quality</small></div><i /><div className="current"><span>CURRENT t</span><strong>Known features</strong><small>no future data</small></div><i /><div><span>TARGET</span><strong>t + {horizon}</strong><small>PM2.5 + event label</small></div></div>
    <div className="grid-two">
      <Panel eyebrow="Sample validity" title="Why timestamps are excluded"><Chart data={[{ type: 'bar', x: ['Valid', 'History', 'Future gap', 'Missing target', 'Quality'], y: [total.valid_training_samples || 0, total.invalid_due_to_history || 0, total.invalid_due_to_future_gap || 0, total.invalid_due_to_missing_target || 0, total.invalid_due_to_quality_issue || 0], marker: { color: ['#0fa97b', '#f0a43b', '#e05f5f', '#b85dc5', '#4779d8'] }, hovertemplate: '%{x}<br>%{y:,}<extra></extra>' }]} layout={{ height: 340, yaxis: { title: { text: 'Samples' } } }} /></Panel>
      <Panel eyebrow="Future target" title={`PM2.5 concentration at t + ${horizon}`}><Chart data={bySensor.map((group: Row[], index) => ({ type: 'scattergl', mode: 'lines', x: group.map(row => row.timestamp), y: group.map(row => row[`target_pm25_${horizon}`]), name: `Sensor ${group[0].sensor_id}`, line: { color: COLORS[index], width: 1.5 }, connectgaps: false }))} layout={{ height: 340, yaxis: { title: { text: 'Future PM2.5 (ug/m3)' } } }} /></Panel>
    </div>
    <Panel eyebrow="Per-sensor readiness" title="Valid samples and gap impact"><DataTable rows={validity.items || []} columns={[["sensor_id", "Sensor"], ["city", "City"], ["horizon", "Horizon"], ["total_timestamps", "Total"], ["valid_training_samples", "Valid"], ["invalid_due_to_history", "History"], ["invalid_due_to_future_gap", "Future gap"], ["invalid_due_to_missing_target", "Missing target"], ["invalid_due_to_quality_issue", "Quality"], ["valid_event_labels", "Event labels"], ["positive_event_labels", "Positive events"]]} /></Panel>
    <Notice title="Recommended Phase 4 comparison" text="Start with per-sensor PM2.5 baselines, then compare a combined sensor-aware model using chronological walk-forward validation. Fit scaling, imputation, and feature selection on training folds only." />
    <div className="download-row"><a className="button primary" href={downloadUrl('/api/ml/download/AirAware_ML_Features.csv')}><Download size={15} /> ML Features</a><a className="button" href={downloadUrl('/api/ml/download/AirAware_Forecast_Targets.csv')}><Download size={15} /> Forecast Targets</a><a className="button" href={downloadUrl('/api/ml/download/forecast_sample_validity.csv')}><Download size={15} /> Validity CSV</a></div>
  </div>
}

export default function Phase3Page({ view, filters }: { view: Phase3View; filters: Filters }) {
  return <div className="phase3-shell"><div className="phase3-heading"><div>{view === 'features' ? <FlaskConical /> : view === 'who' ? <HeartPulse /> : <Route />}<span>AirAware Phase 3</span><h2>{view === 'features' ? 'ML-focused Feature Engineering' : view === 'who' ? 'WHO 2021 Health Analysis' : 'Forecast Dataset Preparation'}</h2></div><div className="phase-chip"><CheckCircle2 /> Local cached outputs</div></div>{view === 'features' ? <FeatureEngineering filters={filters} /> : view === 'who' ? <WhoHealth filters={filters} /> : <ForecastDataset filters={filters} />}</div>
}
