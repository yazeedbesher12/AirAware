import { Info } from 'lucide-react'

const number = (value: number | null) => value == null || !Number.isFinite(value) ? 'Not available' : value.toFixed(1)

export default function SafetyUpperEstimate({ expected, upper, margin }: { expected: number | null; upper: number | null; margin: number | null }) {
  const available = expected != null && upper != null && Number.isFinite(expected) && Number.isFinite(upper)
  const expectedPosition = available ? Math.min(92, Math.max(8, (expected / Math.max(expected, upper)) * 100)) : 0

  return <section className="safety-estimate" aria-label="Safety upper estimate">
    <div className="details-section-title">
      <span>Forecast uncertainty</span>
      <span className="help-tip" title="An empirical upper estimate used to account for possible model underprediction."><Info />Why this matters</span>
    </div>
    {available ? <>
      <div className="safety-values">
        <span><small>Expected</small><strong>{number(expected)}</strong><em>µg/m³</em></span>
        <span><small>Safety Upper Estimate</small><strong>{number(upper)}</strong><em>µg/m³</em></span>
      </div>
      <div className="safety-track" aria-hidden="true">
        <span className="expected-point" style={{ left: `${expectedPosition}%` }} />
        <span className="upper-point" />
      </div>
      <p>Allows for <b>+{number(margin)} µg/m³</b> of possible model underprediction.</p>
    </> : <div className="inline-empty">Forecast unavailable</div>}
  </section>
}
