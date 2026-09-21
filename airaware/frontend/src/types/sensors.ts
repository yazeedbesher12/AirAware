export type RiskLevel = 'Normal' | 'Elevated' | 'High'
export type TimelineMode = 'current' | '1h' | '3h' | '6h'
export type ForecastHorizon = Exclude<TimelineMode, 'current'>

export type EpaCategory =
  | 'Good'
  | 'Moderate'
  | 'Unhealthy for Sensitive Groups'
  | 'Unhealthy'
  | 'Very Unhealthy'
  | 'Hazardous'

export interface SensorCoordinates {
  latitude: number
  longitude: number
}

export interface SensorForecast {
  pm25: number | null
  safetyUpper: number | null
  safetyMargin: number | null
  future24hAverage: number | null
  highPmRisk: RiskLevel | null
  epaAqi: number | null
  epaCategory: EpaCategory | null
  whoStatus: 'Within WHO Guideline' | 'Above WHO Guideline' | null
  whoGuideline: number
  whoRatio: number | null
}

export interface SensorData {
  id: number
  name: string
  city: 'Tulkarem' | 'Nablus'
  site: string
  siteFullName?: string
  coordinates: SensorCoordinates
  coordinatesAreSiteLevel: true
  online: boolean
  lastUpdated: string
  current: {
    pm25: number | null
    temperature: number | null
    humidity: number | null
  }
  currentHealth: {
    epaAqi: number | null
    epaCategory: EpaCategory | null
    whoStatus: 'Within WHO Guideline' | 'Above WHO Guideline' | null
    highPmRisk: RiskLevel | null
  }
  forecasts: Record<ForecastHorizon, SensorForecast>
}

export interface SensorSnapshot {
  pm25: number | null
  epaAqi: number | null
  epaCategory: EpaCategory | null
  whoStatus: SensorData['currentHealth']['whoStatus']
  highPmRisk: RiskLevel | null
  safetyUpper: number | null
  safetyMargin: number | null
  future24hAverage: number | null
  whoGuideline: number | null
  whoRatio: number | null
}

// Mirrors the future ForecastEngine response so the service adapter, rather
// than the map UI, absorbs the backend naming and nesting conventions.
export interface ForecastEngineHorizon {
  pm25: { forecast: number; safety_upper_estimate: number; safety_margin: number }
  future_24h_average: { forecast: number }
  high_pm_risk: { level: RiskLevel }
  who: {
    status: SensorForecast['whoStatus']
    guideline: number
    ratio_to_guideline: number
    averaging_period: '24h'
  }
  epa_aqi: {
    value: number
    category: EpaCategory
    basis: 'PM2.5'
    overall_aqi: false
  }
}
