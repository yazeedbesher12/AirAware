import { Clock3, Droplets, MapPin, Thermometer, Wind, X } from 'lucide-react'
import ForecastMiniChart from './ForecastMiniChart'
import HealthInterpretation from './HealthInterpretation'
import RiskBadge from './RiskBadge'
import SafetyUpperEstimate from './SafetyUpperEstimate'
import type { ForecastHorizon, SensorData, SensorSnapshot, TimelineMode } from '../../types/sensors'

const modeName: Record<TimelineMode, string> = { current: 'Current Conditions', '1h': '1 Hour Forecast', '3h': '3 Hour Forecast', '6h': '6 Hour Forecast' }
const shortTime = (value: string) => new Date(value).toLocaleString(undefined, {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short',
})
const number = (value: number | null, digits = 1) => value == null || !Number.isFinite(value) ? 'Not available' : value.toFixed(digits)

function Condition({ icon, label, value, unit }: { icon: React.ReactNode; label: string; value: number | null; unit: string }) {
  return <span>{icon}<small>{label}</small><b className={value == null ? 'missing' : ''}>{number(value)} {value != null && <em>{unit}</em>}</b></span>
}

export default function SensorDetails({ sensor, snapshot, timeline, onClose }: {
  sensor: SensorData
  snapshot: SensorSnapshot
  timeline: TimelineMode
  onClose: () => void
}) {
  return <aside className="sensor-details" aria-label={`${sensor.name} details`}>
    <div className="sensor-details-head">
      <span className={`sensor-state ${sensor.online ? 'online' : 'offline'}`}><i />{sensor.online ? 'Online' : 'Offline'}</span>
      <button type="button" onClick={onClose} aria-label="Close sensor details"><X /></button>
      <h3>{sensor.name}</h3>
      <h4 className="sensor-site" title={sensor.siteFullName}>{sensor.siteFullName ?? sensor.site}</h4>
      <p><MapPin />{sensor.city}, Palestine</p>
      <small><Clock3 /> Updated {shortTime(sensor.lastUpdated)}</small>
    </div>

    {!sensor.online && <div className="offline-notice">Sensor offline · showing its last available reading and forecast snapshot.</div>}

    <section className="selected-outlook">
      <div className="outlook-heading"><span>{modeName[timeline]}</span><RiskBadge level={snapshot.highPmRisk} /></div>
      <div className="outlook-primary"><strong>{number(snapshot.pm25)}</strong>{snapshot.pm25 != null && <span>µg/m³</span>}<small>PM2.5 {timeline === 'current' ? 'concentration' : 'forecast'}</small></div>
      {timeline === 'current'
        ? <p className="outlook-context">Latest sensor conditions and available health interpretation.</p>
        : <SafetyUpperEstimate expected={snapshot.pm25} upper={snapshot.safetyUpper} margin={snapshot.safetyMargin} />}
    </section>

    <section className="current-conditions">
      <div className="details-section-title"><span>Current conditions</span><small>Latest sensor reading</small></div>
      <div className="conditions-grid">
        <Condition icon={<Wind />} label="PM2.5" value={sensor.current.pm25} unit="µg/m³" />
        <Condition icon={<Thermometer />} label="Temperature" value={sensor.current.temperature} unit="°C" />
        <Condition icon={<Droplets />} label="Humidity" value={sensor.current.humidity} unit="%" />
      </div>
    </section>

    <HealthInterpretation snapshot={snapshot} timeline={timeline} />
    <ForecastMiniChart sensor={sensor} timeline={timeline} />

    <section className="forecast-comparison">
      <div className="details-section-title"><span>Forecast comparison</span><small>Expected PM2.5</small></div>
      <div className="forecast-table" role="table" aria-label="Forecast comparison">
        {(['1h', '3h', '6h'] as ForecastHorizon[]).map(horizon => {
          const forecast = sensor.forecasts[horizon]
          return <div className={timeline === horizon ? 'active' : ''} role="row" key={horizon}>
            <b>{horizon.toUpperCase()}</b>
            <span><small>PM2.5</small>{number(forecast.pm25)}{forecast.pm25 != null && <em> µg/m³</em>}</span>
            <span><small>Safety Upper</small>{number(forecast.safetyUpper)}{forecast.safetyUpper != null && <em> µg/m³</em>}</span>
            <RiskBadge level={forecast.highPmRisk} compact />
          </div>
        })}
      </div>
    </section>
  </aside>
}
