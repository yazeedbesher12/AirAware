const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export type Filters = {
  city: string
  sensorId: string
  feature: string
  startTime: string
  endTime: string
}

export async function api<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function query(params: Record<string, string | number | undefined>): string {
  const output = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== '') output.set(key, String(value))
  })
  const text = output.toString()
  return text ? `?${text}` : ''
}

export const downloadUrl = (path: string) => `${API_BASE}${path}`

