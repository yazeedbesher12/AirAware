import { useEffect, useState } from 'react'
import { getSensors } from '../services/sensors'
import type { SensorData } from '../types/sensors'

export function useSensors() {
  const [sensors, setSensors] = useState<SensorData[]>([])
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    getSensors()
      .then(data => {
        if (!active) return
        setSensors(data)
        setStatus('ready')
      })
      .catch(reason => {
        if (!active) return
        setError(reason instanceof Error ? reason.message : 'Sensor data could not be loaded.')
        setStatus('error')
      })
    return () => { active = false }
  }, [])

  return { sensors, status, error }
}
