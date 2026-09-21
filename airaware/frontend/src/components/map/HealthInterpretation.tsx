import { HeartPulse, ShieldCheck } from 'lucide-react'
import type { SensorSnapshot, TimelineMode } from '../../types/sensors'

const number = (value: number | null, digits = 1) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)

export default function HealthInterpretation({ snapshot, timeline }: { snapshot: SensorSnapshot; timeline: TimelineMode }) {
  const forecastMode = timeline !== 'current'
  const whoLabel = snapshot.whoStatus === 'Within WHO Guideline' ? 'Meets WHO Guideline' : snapshot.whoStatus

  return <section className="health-section">
    <div className="details-section-title"><span>Health interpretation</span><small>{forecastMode ? 'Forecast health context' : 'Latest available'}</small></div>
    <div className="health-cards">
      <article className={`who-card ${snapshot.whoStatus === 'Above WHO Guideline' ? 'above' : ''}`}>
        <div className="health-card-label"><HeartPulse /><span>WHO PM2.5 · 24h</span></div>
        {snapshot.future24hAverage != null ? <>
          <div className="who-status-row"><strong>{number(snapshot.future24hAverage)}</strong><em>µg/m³ predicted average</em></div>
          <b>{whoLabel}</b>
          <div className="who-reference"><span>Guideline <strong>{number(snapshot.whoGuideline, 0)} µg/m³</strong></span><span>Ratio <strong>{number(snapshot.whoRatio, 2)}×</strong></span></div>
        </> : <div className="health-unavailable"><strong>{whoLabel ?? 'Not available'}</strong><span>WHO interpretation requires a predicted PM2.5 24h average.</span></div>}
      </article>

      <article className={`epa-card aqi-${snapshot.epaCategory?.toLowerCase().replaceAll(' ', '-') ?? 'unavailable'}`}>
        <div className="health-card-label"><ShieldCheck /><span>PM2.5-derived EPA AQI</span></div>
        {snapshot.epaAqi != null ? <>
          <div className="aqi-value-row"><strong>{snapshot.epaAqi}</strong><b>{snapshot.epaCategory}</b></div>
          <p>{forecastMode ? 'Based on predicted PM2.5 24h average' : 'Based on the latest PM2.5 reading'}</p>
        </> : <div className="health-unavailable"><strong>Not available</strong><span>No PM2.5-derived EPA AQI is available for this reading.</span></div>}
      </article>
    </div>
  </section>
}
