import { mockSensors } from '../data/sensorsMock'
import type { ForecastEngineHorizon, SensorData, SensorForecast, SensorSnapshot, TimelineMode } from '../types/sensors'

export function adaptForecastHorizon(source: ForecastEngineHorizon): SensorForecast {
  return {
    pm25: source.pm25.forecast,
    safetyUpper: source.pm25.safety_upper_estimate,
    safetyMargin: source.pm25.safety_margin,
    future24hAverage: source.future_24h_average.forecast,
    highPmRisk: source.high_pm_risk.level,
    epaAqi: source.epa_aqi.value,
    epaCategory: source.epa_aqi.category,
    whoStatus: source.who.status,
    whoGuideline: source.who.guideline,
    whoRatio: source.who.ratio_to_guideline,
  }
}

export function getSensorSnapshot(sensor: SensorData, timeline: TimelineMode): SensorSnapshot {
  if (timeline === 'current') {
    return {
      pm25: sensor.current.pm25,
      ...sensor.currentHealth,
      safetyUpper: null,
      safetyMargin: null,
      future24hAverage: null,
      whoGuideline: null,
      whoRatio: null,
    }
  }

  return sensor.forecasts[timeline]
}

export async function getSensors(): Promise<SensorData[]> {
  // Future integration point: replace this source with GET /api/sensors or a
  // realtime subscription, adapt its payload here, and leave the map untouched.
  return Promise.resolve(mockSensors)
}
