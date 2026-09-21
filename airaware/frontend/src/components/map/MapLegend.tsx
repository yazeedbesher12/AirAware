import { Info } from 'lucide-react'

export default function MapLegend() {
  return <aside className="map-legend" aria-label="Map legend">
    <div className="legend-heading">AirAware Risk</div>
    <div className="risk-key"><span><i className="risk-dot risk-normal" />Normal</span><span><i className="risk-dot risk-elevated" />Elevated</span><span><i className="risk-dot risk-high" />High</span></div>
    <p><Info /> Marker color shows the separate PM2.5-derived EPA category.</p>
  </aside>
}
