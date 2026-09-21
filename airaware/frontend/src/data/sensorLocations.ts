import type { SensorCoordinates } from '../types/sensors'

/**
 * Verified deployment-site coordinates. Replace with exact physical sensor
 * GPS coordinates if device-level coordinates become available.
 */
export const SENSOR_LOCATIONS: Record<number, SensorCoordinates> = {
  1: { latitude: 32.3096125, longitude: 35.0251874 },
  2: { latitude: 32.22819, longitude: 35.22224 },
  4: { latitude: 32.22025, longitude: 35.24431 },
  5: { latitude: 32.21575, longitude: 35.29659 },
}
