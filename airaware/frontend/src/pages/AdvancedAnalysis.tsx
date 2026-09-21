import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, BarChart3, BrainCircuit, Clock3, Download, GitCompareArrows, Layers3, Network, ScanSearch, Split, TrendingUp } from 'lucide-react'
import Chart from '../components/Chart'
import { api, downloadUrl, query, type Filters } from '../services/api'

type Row = Record<string, any>
type Tab = 'statistics' | 'temporal' | 'correlation' | 'lag' | 'relationships' | 'pca' | 'clusters' | 'anomalies' | 'changes'

const COLORS = ['#0fa97b', '#4779d8', '#f0a43b', '#b85dc5', '#e05f5f', '#667a72']
const FEATURE_LABELS: Record<string, string> = { pm25: 'PM2.5', temperature: 'Temperature', humidity: 'Humidity', no2: 'NO₂', o3: 'O₃', aqi: 'Existing AQI' }
const fmt = (value: unknown, digits = 2) => value == null || value === '' || (typeof value === 'number' && !Number.isFinite(value)) ? '—' : typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : String(value)
const when = (value: unknown) => value ? new Date(String(value)).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }) : '—'
const featureLabel = (value: string) => FEATURE_LABELS[value] || value
const groupRows = <T,>(items: T[], key: (item: T) => string) => items.reduce<Record<string, T[]>>((groups, item) => { const name = key(item); (groups[name] ||= []).push(item); return groups }, {})

function Panel({ eyebrow, title, children, action }: { eyebrow: string; title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return <section className="card"><div className="card-head"><div><span className="eyebrow">{eyebrow}</span><h3>{title}</h3></div>{action}</div>{children}</section>
}

function Findings({ items, warning }: { items: string[]; warning?: string }) {
  return <aside className="findings-panel"><div className="findings-title"><BrainCircuit /><div><span>Evidence panel</span><strong>Key Findings</strong></div></div><ul>{items.filter(Boolean).slice(0, 6).map((item, index) => <li key={index}>{item}</li>)}</ul>{warning && <p><AlertTriangle />{warning}</p>}</aside>
}

function Empty({ text }: { text: string }) {
  return <div className="advanced-empty"><ScanSearch /><strong>No valid analysis for this selection</strong><span>{text}</span></div>
}

function DataTable({ columns, rows, limit = 100 }: { columns: [string, string][]; rows: Row[]; limit?: number }) {
  return <div className="table-wrap dense"><table><thead><tr>{columns.map(([, label]) => <th key={label}>{label}</th>)}</tr></thead><tbody>{rows.slice(0, limit).map((row, index) => <tr key={index}>{columns.map(([key]) => <td key={key}>{typeof row[key] === 'number' ? fmt(row[key], 3) : key.includes('timestamp') ? when(row[key]) : fmt(row[key])}</td>)}</tr>)}</tbody></table></div>
}

function StatisticalPage({ statistics, distribution, feature }: { statistics: Row; distribution: Row; feature: string }) {
  const rows = statistics.items || []
  const series = distribution.series || []
  const interpretations = (statistics.interpretations || []).map((item: Row) => item.text)
  if (!distribution.available) return <><Panel eyebrow="Distribution availability" title={featureLabel(feature)}><Empty text={distribution.message || 'The selected sensors do not provide this feature.'} /></Panel><Findings items={[distribution.message]} /></>
  return <div className="advanced-stack">
    <div className="grid-two">
      <Panel eyebrow="Distribution" title={`${featureLabel(feature)} histogram`}><Chart data={series.map((s: Row, i: number) => ({ type: 'histogram', x: s.sample, name: `#${s.sensor_id}`, opacity: .65, marker: { color: COLORS[i] }, nbinsx: 36 }))} layout={{ barmode: 'overlay', height: 330, xaxis: { title: { text: `${featureLabel(feature)} (${distribution.unit})` } } }} /></Panel>
      <Panel eyebrow="Distribution" title="Boxplot by sensor"><Chart data={series.map((s: Row, i: number) => ({ type: 'box', y: s.sample, name: `#${s.sensor_id}`, marker: { color: COLORS[i] }, boxpoints: 'outliers' }))} layout={{ height: 330, yaxis: { title: { text: `${featureLabel(feature)} (${distribution.unit})` } } }} /></Panel>
    </div>
    <div className="grid-two">
      <Panel eyebrow="Cumulative distribution" title="Empirical CDF"><Chart data={series.map((s: Row, i: number) => ({ type: 'scatter', mode: 'lines', x: s.ecdf.x, y: s.ecdf.y, name: `#${s.sensor_id}`, line: { color: COLORS[i], width: 2 } }))} layout={{ height: 330, xaxis: { title: { text: featureLabel(feature) } }, yaxis: { title: { text: 'Cumulative probability' }, range: [0, 1] } }} /></Panel>
      <Panel eyebrow="Density" title="Kernel density estimate"><Chart data={series.filter((s: Row) => s.kde.x.length).map((s: Row, i: number) => ({ type: 'scatter', mode: 'lines', x: s.kde.x, y: s.kde.density, fill: 'tozeroy', name: `#${s.sensor_id}`, line: { color: COLORS[i] } }))} layout={{ height: 330, xaxis: { title: { text: featureLabel(feature) } }, yaxis: { title: { text: 'Density' } } }} /></Panel>
    </div>
    <Panel eyebrow="Descriptive statistics" title="Percentiles, shape, and variability"><DataTable rows={rows} columns={[["scope_value", "Scope"], ["count", "Count"], ["mean", "Mean"], ["median", "Median"], ["std", "Std"], ["variance", "Variance"], ["q1", "Q1"], ["q3", "Q3"], ["iqr", "IQR"], ["p01", "P01"], ["p05", "P05"], ["p95", "P95"], ["p99", "P99"], ["skewness", "Skew"], ["kurtosis", "Kurtosis"], ["coefficient_of_variation", "CV"]]} /></Panel>
    <Findings items={interpretations.length ? interpretations : ['Distribution statistics are available, with no automatic invalidation of extreme tails.']} />
  </div>
}

function TemporalPage({ temporal, rolling, decomposition, feature, window, setWindow }: { temporal: Row; rolling: Row; decomposition: Row; feature: string; window: string; setWindow: (value: string) => void }) {
  const items = temporal.items || []
  const byType = (type: string) => items.filter((row: Row) => row.pattern_type === type)
  const hours = byType('hour'), days = byType('day_of_week'), daily = byType('daily')
  const findings = [...(temporal.findings || []).map((x: Row) => x.text), ...(decomposition.findings || []).map((x: Row) => x.text)]
  const decompositionSeries = (decomposition.items || []).filter((x: Row) => x.status === 'AVAILABLE')
  return <div className="advanced-stack">
    <Panel eyebrow="Rolling statistics" title={`${featureLabel(feature)} raw and rolling behavior`} action={<label className="inline-control"><span>Window</span><select value={window} onChange={e => setWindow(e.target.value)}>{['1h', '3h', '6h', '12h', '24h'].map(x => <option key={x}>{x}</option>)}</select></label>}>
      {rolling.available ? <Chart data={(rolling.series || []).flatMap((s: Row, i: number) => [{ type: 'scattergl', mode: 'lines', x: s.points.map((p: Row) => p.timestamp), y: s.points.map((p: Row) => p.raw), name: `#${s.sensor_id} raw`, line: { color: COLORS[i], width: 1 } }, { type: 'scattergl', mode: 'lines', x: s.points.map((p: Row) => p.timestamp), y: s.points.map((p: Row) => p.mean), name: `#${s.sensor_id} ${window} mean`, line: { color: COLORS[i], width: 2.4, dash: 'dot' } }])} layout={{ height: 420, xaxis: { rangeslider: { visible: true, thickness: .07 } }, yaxis: { title: { text: `${featureLabel(feature)} (${rolling.unit})` } } }} /> : <Empty text={`${featureLabel(feature)} is unavailable for this sensor selection.`} />}
    </Panel>
    <div className="grid-two"><Panel eyebrow="Diurnal profile" title="Average and variability by hour"><Chart data={[{ type: 'scatter', mode: 'lines+markers', x: hours.map((r: Row) => r.period), y: hours.map((r: Row) => r.mean), error_y: { type: 'data', array: hours.map((r: Row) => r.std), visible: true, thickness: .7 }, marker: { color: '#0fa97b' }, line: { color: '#0fa97b' } }]} layout={{ height: 330, xaxis: { title: { text: 'UTC hour' }, dtick: 2 }, yaxis: { title: { text: 'Mean ± std' } } }} /></Panel>
      <Panel eyebrow="Weekly profile" title="Day-of-week pattern"><Chart data={[{ type: 'bar', x: days.map((r: Row) => r.period), y: days.map((r: Row) => r.mean), marker: { color: '#4779d8' }, error_y: { type: 'data', array: days.map((r: Row) => r.std), visible: true } }]} layout={{ height: 330, yaxis: { title: { text: 'Mean ± std' } } }} /></Panel></div>
    <Panel eyebrow="Daily trend" title="Daily mean, median, and maximum"><Chart data={['mean', 'median', 'max'].map((metric, i) => ({ type: 'scatter', mode: 'lines+markers', x: daily.map((r: Row) => r.period), y: daily.map((r: Row) => r[metric]), name: metric, line: { color: COLORS[i], width: 2 } }))} layout={{ height: 350 }} /></Panel>
    <Panel eyebrow="STL decomposition" title="Observed, trend, daily seasonality, and residual">{decompositionSeries.length ? <Chart data={decompositionSeries.flatMap((s: Row, i: number) => ['observed', 'trend', 'seasonal', 'residual'].map((key, j) => ({ type: 'scattergl', mode: 'lines', x: s.points.map((p: Row) => p.timestamp), y: s.points.map((p: Row) => p[key]), name: `#${s.sensor_id} ${key}`, line: { color: COLORS[(i + j) % COLORS.length], width: key === 'observed' ? 1 : 1.7 } })))} layout={{ height: 430 }} /> : <Empty text={(decomposition.items || [])[0]?.reason || 'No continuous segment passed the decomposition requirements.'} />}</Panel>
    <Findings items={findings.length ? findings : ['Temporal profiles use each sensor independently and retain gaps as missing time positions.']} />
  </div>
}

function CorrelationPage({ pearson, spearman, scatter, feature, featureB, setFeatureB }: { pearson: Row; spearman: Row; scatter: Row; feature: string; featureB: string; setFeatureB: (v: string) => void }) {
  const strongest = (pearson.relationships || []).slice(0, 6)
  const nonlinear = (spearman.nonlinear_candidates || []).slice(0, 4)
  const findings = [...strongest.map((r: Row) => `${featureLabel(r.feature_a)} and ${featureLabel(r.feature_b)}: Pearson ${fmt(r.correlation, 3)} (n=${fmt(r.n_observations, 0)}).`), ...nonlinear.map((r: Row) => `${featureLabel(r.feature_a)} ↔ ${featureLabel(r.feature_b)} has stronger rank than linear association; no causation is implied.`)]
  const heat = (payload: Row, title: string) => <Panel eyebrow="Dependency matrix" title={title}>{payload.features?.length ? <Chart data={[{ type: 'heatmap', x: payload.features.map(featureLabel), y: payload.features.map(featureLabel), z: payload.matrix, zmin: -1, zmax: 1, colorscale: [[0, '#d85e5e'], [.5, '#f4f6f5'], [1, '#0fa97b']], text: payload.matrix.map((row: any[]) => row.map(v => v == null ? '—' : Number(v).toFixed(2))), texttemplate: '%{text}' }]} layout={{ height: 390 }} /> : <Empty text="No complete feature pairs exist for this selection." />}</Panel>
  return <div className="advanced-stack">
    <div className="callout info"><Network /><div><strong>How to read the correlation matrices</strong><p>Pearson measures linear movement from -1 to +1: values near +1 move together, values near -1 move in opposite directions, and values near 0 show little linear relationship. Spearman compares rank order and can reveal monotonic nonlinear relationships. Neither matrix proves causation.</p></div></div>
    <div className="grid-two correlation-matrices">{heat(pearson, 'Pearson correlation')}{heat(spearman, 'Spearman correlation')}</div>
    <Panel eyebrow="Feature-pair explorer" title={`${featureLabel(feature)} relationship`} action={<label className="inline-control"><span>Compare with</span><select value={featureB} onChange={e => setFeatureB(e.target.value)}>{Object.entries(FEATURE_LABELS).filter(([key]) => key !== feature).map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>}>
      {scatter.available ? <Chart data={scatter.series.map((s: Row, i: number) => ({ type: 'scattergl', mode: 'markers', x: s.points.map((p: Row) => p.x), y: s.points.map((p: Row) => p.y), name: `#${s.sensor_id}`, marker: { color: COLORS[i], opacity: .5, size: 5 } }))} layout={{ height: 400, hovermode: 'closest', xaxis: { title: { text: featureLabel(feature) } }, yaxis: { title: { text: featureLabel(featureB) } } }} /> : <Empty text="One or both selected features are unavailable together." />}
    </Panel>
    <Panel eyebrow="Ranked relationships" title="Strongest positive and negative relationships"><DataTable rows={(pearson.relationships || []).slice(0, 30)} columns={[["scope_value", "Scope"], ["feature_a", "Feature A"], ["feature_b", "Feature B"], ["correlation", "Pearson"], ["n_observations", "Overlap"]]} /></Panel>
    <Findings items={findings} warning="Correlation and mutual movement do not establish causation." />
  </div>
}

function LagPage({ autocorrelation, cross }: { autocorrelation: Row; cross: Row }) {
  const auto = autocorrelation.items || [], relationships = cross.items || []
  const top = relationships.filter((r: Row) => r.lag_minutes !== 0).slice(0, 8)
  const findings = [...auto.slice(0, 4).map((r: Row) => `Sensor ${r.sensor_id} ${featureLabel(r.feature)} retains autocorrelation ${fmt(r.autocorrelation, 3)} at ${r.lag}.`), ...top.slice(0, 3).map((r: Row) => `${featureLabel(r.source_feature)}(t) is statistically associated with ${featureLabel(r.target_feature)}(t${r.lag}); r=${fmt(r.correlation, 3)}.`)]
  return <div className="advanced-stack"><div className="callout info"><Clock3 /><div><strong>Elapsed-time lags</strong><p>{autocorrelation.methodology}</p></div></div>
    <div className="grid-two"><Panel eyebrow="Serial dependency" title="ACF by elapsed lag"><Chart data={Object.entries(groupRows(auto, (r: any) => `Sensor ${r.sensor_id} · ${featureLabel(r.feature)}`)).map(([name, rows], i) => ({ type: 'scatter', mode: 'lines+markers', name, x: rows.map((r: any) => r.lag_minutes / 60), y: rows.map((r: any) => r.autocorrelation), line: { color: COLORS[i % COLORS.length] } }))} layout={{ height: 360, xaxis: { title: { text: 'Lag (hours)' } }, yaxis: { title: { text: 'Autocorrelation' }, range: [-1, 1] } }} /></Panel>
      <Panel eyebrow="Conditional lag" title="PACF by elapsed lag"><Chart data={Object.entries(groupRows(auto, (r: any) => `Sensor ${r.sensor_id} · ${featureLabel(r.feature)}`)).map(([name, rows], i) => ({ type: 'bar', name, x: rows.map((r: any) => r.lag_minutes / 60), y: rows.map((r: any) => r.pacf), marker: { color: COLORS[i % COLORS.length] } }))} layout={{ height: 360, barmode: 'group', xaxis: { title: { text: 'Lag (hours)' } }, yaxis: { title: { text: 'Partial autocorrelation' }, range: [-1, 1] } }} /></Panel></div>
    <Panel eyebrow="Cross-correlation" title="Feature relationships across positive and negative lags"><Chart data={Object.entries(groupRows(relationships.slice(0, 120), (r: any) => `${featureLabel(r.source_feature)} → ${featureLabel(r.target_feature)} · #${r.sensor_id}`)).slice(0, 8).map(([name, rows], i) => ({ type: 'scatter', mode: 'lines+markers', name, x: rows.map((r: any) => r.lag_minutes / 60), y: rows.map((r: any) => r.correlation), line: { color: COLORS[i % COLORS.length] } }))} layout={{ height: 410, xaxis: { title: { text: 'Target offset (hours)' } }, yaxis: { title: { text: 'Correlation' }, range: [-1, 1] } }} /></Panel>
    <Panel eyebrow="Forecasting candidates" title="Strongest predictive time relationships"><DataTable rows={top} columns={[["sensor_id", "Sensor"], ["source_feature", "Source"], ["target_feature", "Target"], ["lag", "Lag"], ["correlation", "Correlation"], ["n_observations", "Overlap"]]} /></Panel>
    <Findings items={findings} warning="These are statistical associations, not causal effects or trained forecasting features." />
  </div>
}

function RelationshipsPage({ comparison, timeseries, feature }: { comparison: Row; timeseries: Row; feature: string }) {
  const pairs = comparison.pairs || [], differences = comparison.difference_series || []
  const findings = pairs.slice(0, 6).map((p: Row) => `Sensors ${p.sensor_pair}: Pearson ${fmt(p.pearson, 3)}, Spearman ${fmt(p.spearman, 3)}, MAE ${fmt(p.mae, 2)}, overlap ${fmt(p.overlap_count, 0)}.`)
  return <div className="advanced-stack"><div className="callout info"><GitCompareArrows /><div><strong>Nablus aligned timestamps only</strong><p>Divergence may reflect location-specific conditions and is not automatically classified as sensor failure.</p></div></div>
    <Panel eyebrow="Multi-sensor time series" title={`Nablus ${featureLabel(feature)}`}>{timeseries.available ? <Chart data={timeseries.series.map((s: Row, i: number) => ({ type: 'scattergl', mode: 'lines', x: s.points.map((p: Row) => p.timestamp), y: s.points.map((p: Row) => p.value), name: `#${s.sensor_id}`, line: { color: COLORS[i], width: 1.4 } }))} layout={{ height: 420 }} /> : <Empty text="This feature is not available across Nablus sensors." />}</Panel>
    <div className="grid-two"><Panel eyebrow="Pairwise scatter" title="Synchronized sensor readings">{differences.length ? <Chart data={differences.map((pair: Row, i: number) => ({ type: 'scattergl', mode: 'markers', x: pair.points.map((p: Row) => p.a), y: pair.points.map((p: Row) => p.b), name: pair.sensor_pair, marker: { color: COLORS[i], size: 5, opacity: .5 } }))} layout={{ height: 360, hovermode: 'closest', xaxis: { title: { text: 'Sensor A' } }, yaxis: { title: { text: 'Sensor B' } } }} /> : <Empty text="No pairwise overlap." />}</Panel>
      <Panel eyebrow="Difference over time" title="Sensor A minus Sensor B">{differences.length ? <Chart data={differences.map((pair: Row, i: number) => ({ type: 'scattergl', mode: 'lines', x: pair.points.map((p: Row) => p.timestamp), y: pair.points.map((p: Row) => p.difference), name: pair.sensor_pair, line: { color: COLORS[i] } }))} layout={{ height: 360, yaxis: { title: { text: 'Difference' }, zeroline: true } }} /> : <Empty text="No pairwise overlap." />}</Panel></div>
    <Panel eyebrow="Sensor agreement" title="Pairwise statistics"><DataTable rows={pairs} columns={[["sensor_pair", "Pair"], ["overlap_count", "Overlap"], ["pearson", "Pearson"], ["spearman", "Spearman"], ["mae", "MAE"], ["median_absolute_difference", "Median abs diff"], ["mean_difference_a_minus_b", "Mean A−B"], ["difference_std", "Difference std"]]} /></Panel>
    <Findings items={findings} />
  </div>
}

function PcaPage({ data }: { data: Row }) {
  const group = (data.groups || []).find((x: Row) => x.status === 'AVAILABLE'), variance = data.variance || [], loadings = data.loadings || []
  if (!group) return <Empty text="No city group passed the complete-feature PCA requirements." />
  const components = [...new Set<string>(loadings.map((r: Row) => String(r.component)))]
  const features = [...new Set<string>(loadings.map((r: Row) => String(r.feature)))]
  const matrix = components.map(component => features.map(feature => loadings.find((r: Row) => r.component === component && r.feature === feature)?.loading ?? null))
  return <div className="advanced-stack"><div className="advanced-kpis"><span><small>Group</small><b>{group.group}</b></span><span><small>Valid rows</small><b>{fmt(group.n_observations, 0)}</b></span><span><small>Components for 90%</small><b>{group.components_for_threshold['0.9']}</b></span><span><small>Features</small><b>{group.features.length}</b></span></div>
    <div className="grid-two"><Panel eyebrow="Explained variance" title="Component contribution"><Chart data={[{ type: 'bar', x: variance.map((r: Row) => r.component), y: variance.map((r: Row) => r.explained_variance_ratio), name: 'Individual', marker: { color: '#0fa97b' } }, { type: 'scatter', mode: 'lines+markers', x: variance.map((r: Row) => r.component), y: variance.map((r: Row) => r.cumulative_explained_variance), name: 'Cumulative', yaxis: 'y2', line: { color: '#e59d35' } }]} layout={{ height: 360, yaxis: { title: { text: 'Explained ratio' } }, yaxis2: { title: { text: 'Cumulative' }, overlaying: 'y', side: 'right', range: [0, 1] } }} /></Panel>
      <Panel eyebrow="Component loadings" title="Standardized feature contribution"><Chart data={[{ type: 'heatmap', x: features.map(featureLabel), y: components, z: matrix, zmin: -1, zmax: 1, colorscale: [[0, '#4779d8'], [.5, '#f5f7f6'], [1, '#e15e5e']], text: matrix.map(row => row.map(v => v == null ? '—' : Number(v).toFixed(2))), texttemplate: '%{text}' }]} layout={{ height: 360 }} /></Panel></div>
    <Panel eyebrow="PCA projection" title="First two principal components"><Chart data={Object.entries(groupRows(group.points, (p: any) => `Sensor ${p.sensor_id}`)).map(([name, points], i) => ({ type: 'scattergl', mode: 'markers', x: points.map((p: any) => p.pc1), y: points.map((p: any) => p.pc2), name, marker: { color: COLORS[i], size: 5, opacity: .55 } }))} layout={{ height: 410, hovermode: 'closest', xaxis: { title: { text: 'PC1' } }, yaxis: { title: { text: 'PC2' } } }} /></Panel>
    <Findings items={group.dominant_components.map((x: Row) => x.text)} warning="Components describe statistical co-variation; no real-world pollution source labels are assigned." />
  </div>
}

function ClusterPage({ data }: { data: Row }) {
  const group = (data.groups || []).find((x: Row) => x.status === 'AVAILABLE'), assignment = (data.assignments || [])[0]
  if (!group || !assignment) return <Empty text="No city group passed clustering completeness requirements." />
  const xFeature = assignment.features.includes('pm25') ? 'pm25' : assignment.features[0], yFeature = assignment.features.find((f: string) => f !== xFeature) || assignment.features[1]
  return <div className="advanced-stack"><div className="advanced-kpis"><span><small>Selected method</small><b>{group.method}</b></span><span><small>Selected k</small><b>{group.selected_k}</b></span><span><small>Silhouette</small><b>{fmt(group.silhouette_score, 3)}</b></span><span><small>City group</small><b>{group.group}</b></span></div>
    <div className="grid-two"><Panel eyebrow="Model selection" title="K-Means silhouette by k"><Chart data={[{ type: 'bar', x: group.evaluations.map((e: Row) => e.cluster_count), y: group.evaluations.map((e: Row) => e.silhouette), marker: { color: '#0fa97b' } }]} layout={{ height: 340, xaxis: { title: { text: 'k' }, dtick: 1 }, yaxis: { title: { text: 'Silhouette' } } }} /></Panel>
      <Panel eyebrow="Regime projection" title={`${featureLabel(xFeature)} vs ${featureLabel(yFeature)}`}><Chart data={Object.entries(groupRows(assignment.points, (p: any) => `Cluster ${p.cluster}`)).map(([name, points], i) => ({ type: 'scattergl', mode: 'markers', x: points.map((p: any) => p[xFeature]), y: points.map((p: any) => p[yFeature]), name, marker: { color: COLORS[i], size: 5, opacity: .55 } }))} layout={{ height: 340, hovermode: 'closest', xaxis: { title: { text: featureLabel(xFeature) } }, yaxis: { title: { text: featureLabel(yFeature) } } }} /></Panel></div>
    <Panel eyebrow="Cluster profiles" title="Neutral measurement regimes"><div className="cluster-grid">{group.clusters.map((cluster: Row) => <article key={cluster.cluster}><span>Cluster {cluster.cluster}</span><strong>{fmt(cluster.count, 0)} observations</strong><small>{fmt(cluster.percentage, 1)}% of valid rows</small><p>{cluster.interpretation}</p><em>Typical UTC hours: {cluster.typical_hours.join(', ')}</em></article>)}</div></Panel>
    <Panel eyebrow="Density sensitivity" title="DBSCAN parameter evaluation"><DataTable rows={group.dbscan_sensitivity} columns={[["parameter", "Parameter"], ["cluster_count", "Clusters"], ["noise_count", "Noise points"], ["silhouette", "Silhouette"]]} /></Panel>
    <Findings items={group.clusters.map((cluster: Row) => cluster.interpretation)} warning="Clusters are exploratory regimes, not WHO or health categories." />
  </div>
}

function AnomalyPage({ data, timeseries, feature }: { data: Row; timeseries: Row; feature: string }) {
  const rows = data.items || []
  const bySensor = Object.entries(groupRows(rows, (r: any) => `#${r.sensor_id}`)).map(([key, values]) => [key, values.length])
  const byFeature = Object.entries(groupRows(rows, (r: any) => featureLabel(r.feature))).map(([key, values]) => [key, values.length])
  const findings = [`${fmt(data.total, 0)} anomaly-feature records match the filters; every observation remains in the cleaned dataset.`, `${fmt(data.summary?.consensus_count, 0)} records are supported by at least two methods globally.`, ...Object.entries(data.summary?.category_counts || {}).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${fmt(value, 0)} records.`)]
  return <div className="advanced-stack"><Panel eyebrow="Consensus overlay" title={`${featureLabel(feature)} anomalies over time`}>{timeseries.available ? <Chart data={[...timeseries.series.map((s: Row, i: number) => ({ type: 'scattergl', mode: 'lines', x: s.points.map((p: Row) => p.timestamp), y: s.points.map((p: Row) => p.value), name: `#${s.sensor_id} raw`, line: { color: COLORS[i], width: 1 } })), { type: 'scatter', mode: 'markers', x: rows.filter((r: Row) => r.feature === feature).map((r: Row) => r.timestamp), y: rows.filter((r: Row) => r.feature === feature).map((r: Row) => r.value), name: 'Anomaly flags', marker: { color: '#d84f59', size: rows.filter((r: Row) => r.feature === feature).map((r: Row) => 5 + r.anomaly_methods_triggered * 2), symbol: 'diamond-open' } }]} layout={{ height: 420 }} /> : <Empty text="Feature unavailable." />}</Panel>
    <div className="grid-two"><Panel eyebrow="By sensor" title="Anomaly record count"><Chart data={[{ type: 'bar', x: bySensor.map(x => x[0]), y: bySensor.map(x => x[1]), marker: { color: COLORS } }]} layout={{ height: 320 }} /></Panel><Panel eyebrow="By feature" title="Anomaly record count"><Chart data={[{ type: 'bar', x: byFeature.map(x => x[0]), y: byFeature.map(x => x[1]), marker: { color: '#e09a35' } }]} layout={{ height: 320 }} /></Panel></div>
    <Panel eyebrow="Method agreement" title="Advanced anomaly table" action={<a className="button small" href={downloadUrl('/api/analysis/download/anomaly_results.csv')}><Download size={14} /> CSV</a>}><DataTable rows={rows} columns={[["timestamp", "Timestamp"], ["sensor_id", "Sensor"], ["feature", "Feature"], ["value", "Value"], ["anomaly_methods_triggered", "Methods"], ["methods", "Triggered"], ["quality_flag", "Phase 1 flag"], ["analytical_category", "Context category"]]} /></Panel>
    <Findings items={findings} warning="Consensus flags support investigation only and are never deleted automatically." />
  </div>
}

function ChangePage({ data, timeseries, feature }: { data: Row; timeseries: Row; feature: string }) {
  const rows = data.items || [], selected = rows.filter((r: Row) => r.feature === feature)
  const findings = selected.slice(0, 6).map((r: Row) => `Sensor ${r.sensor_id} ${featureLabel(r.feature)} has a ${r.change_type.toLowerCase().replace('_', ' ')} near ${when(r.change_timestamp)}; magnitude ${fmt(r.magnitude, 3)}.`)
  return <div className="advanced-stack"><Panel eyebrow="Behavioral shifts" title={`${featureLabel(feature)} change points`}>{timeseries.available ? <Chart data={[...timeseries.series.map((s: Row, i: number) => ({ type: 'scattergl', mode: 'lines', x: s.points.map((p: Row) => p.timestamp), y: s.points.map((p: Row) => p.value), name: `#${s.sensor_id}`, line: { color: COLORS[i] } })), ...selected.map((r: Row) => ({ type: 'scatter', mode: 'markers', x: [r.change_timestamp], y: [r.after_statistic], name: `#${r.sensor_id} ${r.change_type}`, marker: { color: '#d94f59', size: 11, symbol: 'x' } }))]} layout={{ height: 430 }} /> : <Empty text="Feature unavailable." />}</Panel>
    <Panel eyebrow="PELT results" title="Before and after statistics" action={<a className="button small" href={downloadUrl('/api/analysis/download/change_points.csv')}><Download size={14} /> CSV</a>}><DataTable rows={rows} columns={[["change_timestamp", "Timestamp"], ["sensor_id", "Sensor"], ["feature", "Feature"], ["change_type", "Type"], ["before_statistic", "Before"], ["after_statistic", "After"], ["magnitude", "Magnitude"], ["label", "Interpretation"]]} /></Panel>
    <Findings items={findings.length ? findings : ['No change point passed the continuity and magnitude requirements for this feature selection.']} warning="Detected shifts are behavioral regime changes, not confirmed calibration drift." />
  </div>
}

export default function AdvancedAnalysis({ filters }: { filters: Filters }) {
  const [tab, setTab] = useState<Tab>('statistics')
  const [data, setData] = useState<Record<string, Row>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [window, setWindow] = useState('3h')
  const [featureB, setFeatureB] = useState(filters.feature === 'humidity' ? 'temperature' : 'humidity')
  const comparisonFeature = ['pm25', 'temperature', 'humidity', 'aqi'].includes(filters.feature) ? filters.feature : 'pm25'
  const params = useMemo(() => ({ city: filters.city, sensor_id: filters.sensorId, feature: filters.feature, start_time: filters.startTime, end_time: filters.endTime ? `${filters.endTime}T23:59:59Z` : '' }), [filters])

  useEffect(() => {
    setLoading(true); setError('')
    const tasks: Promise<Row>[] = []
    if (tab === 'statistics') tasks.push(api(`/api/analysis/statistics${query(params)}`), api(`/api/analysis/distributions${query(params)}`))
    if (tab === 'temporal') tasks.push(api(`/api/analysis/temporal${query(params)}`), api(`/api/analysis/rolling${query({ ...params, window })}`), api(`/api/analysis/decomposition${query(params)}`))
    if (tab === 'correlation') tasks.push(api(`/api/analysis/correlation${query({ ...params, method: 'pearson' })}`), api(`/api/analysis/correlation${query({ ...params, method: 'spearman' })}`), api(`/api/scatter${query({ city: filters.city, sensor_id: filters.sensorId, x_feature: filters.feature, y_feature: featureB })}`))
    if (tab === 'lag') tasks.push(api(`/api/analysis/autocorrelation${query(params)}`), api(`/api/analysis/cross-correlation${query({ ...params, limit: 500 })}`))
    if (tab === 'relationships') tasks.push(api(`/api/analysis/sensor-comparison${query({ feature: comparisonFeature })}`), api(`/api/timeseries${query({ city: 'Nablus', feature: comparisonFeature, max_points: 1400 })}`))
    if (tab === 'pca') tasks.push(api(`/api/analysis/pca${query({ city: filters.city })}`))
    if (tab === 'clusters') tasks.push(api(`/api/analysis/clusters${query({ city: filters.city })}`))
    if (tab === 'anomalies') tasks.push(api(`/api/analysis/anomalies${query({ ...params, limit: 1500 })}`), api(`/api/timeseries${query({ ...params, max_points: 1400 })}`))
    if (tab === 'changes') tasks.push(api(`/api/analysis/change-points${query(params)}`), api(`/api/timeseries${query({ ...params, max_points: 1400 })}`))
    Promise.all(tasks).then(results => {
      const keys: Record<Tab, string[]> = { statistics: ['statistics', 'distribution'], temporal: ['temporal', 'rolling', 'decomposition'], correlation: ['pearson', 'spearman', 'scatter'], lag: ['autocorrelation', 'cross'], relationships: ['comparison', 'timeseries'], pca: ['pca'], clusters: ['clusters'], anomalies: ['anomalies', 'timeseries'], changes: ['changes', 'timeseries'] }
      setData(Object.fromEntries(keys[tab].map((key, i) => [key, results[i]])))
    }).catch(err => setError(err.message)).finally(() => setLoading(false))
  }, [tab, params, window, featureB, comparisonFeature, filters.city, filters.sensorId, filters.feature])

  const tabs: [Tab, React.ElementType, string][] = [['statistics', BarChart3, 'Statistical Overview'], ['temporal', TrendingUp, 'Temporal Patterns'], ['correlation', Network, 'Correlation'], ['lag', Clock3, 'Lag Analysis'], ['relationships', GitCompareArrows, 'Sensor Relationships'], ['pca', Split, 'PCA'], ['clusters', Layers3, 'Clusters'], ['anomalies', ScanSearch, 'Anomalies'], ['changes', Activity, 'Change Points']]
  return <div className="advanced-shell"><div className="advanced-heading"><div><span className="eyebrow">Phase 2 · cached local analysis</span><h2>Advanced Analysis</h2><p>Temporal dependency, multivariate structure, regimes, and anomaly consensus—without forecasting or AQI recalculation.</p></div><span className="phase-chip"><BrainCircuit /> Phase 2</span></div>
    <div className="advanced-tabs">{tabs.map(([key, Icon, text]) => <button className={tab === key ? 'active' : ''} key={key} onClick={() => setTab(key)}><Icon />{text}</button>)}</div>
    {loading && <div className="advanced-loading"><span /><p>Loading cached Phase 2 analysis…</p></div>}
    {error && <div className="aqi-warning"><AlertTriangle /><div><strong>Phase 2 analysis unavailable</strong><p>{error}</p></div></div>}
    {!loading && !error && <>
      {tab === 'statistics' && <StatisticalPage statistics={data.statistics || {}} distribution={data.distribution || {}} feature={filters.feature} />}
      {tab === 'temporal' && <TemporalPage temporal={data.temporal || {}} rolling={data.rolling || {}} decomposition={data.decomposition || {}} feature={filters.feature} window={window} setWindow={setWindow} />}
      {tab === 'correlation' && <CorrelationPage pearson={data.pearson || {}} spearman={data.spearman || {}} scatter={data.scatter || {}} feature={filters.feature} featureB={featureB} setFeatureB={setFeatureB} />}
      {tab === 'lag' && <LagPage autocorrelation={data.autocorrelation || {}} cross={data.cross || {}} />}
      {tab === 'relationships' && <RelationshipsPage comparison={data.comparison || {}} timeseries={data.timeseries || {}} feature={comparisonFeature} />}
      {tab === 'pca' && <PcaPage data={data.pca || {}} />}
      {tab === 'clusters' && <ClusterPage data={data.clusters || {}} />}
      {tab === 'anomalies' && <AnomalyPage data={data.anomalies || {}} timeseries={data.timeseries || {}} feature={filters.feature} />}
      {tab === 'changes' && <ChangePage data={data.changes || {}} timeseries={data.timeseries || {}} feature={filters.feature} />}
    </>}
  </div>
}
