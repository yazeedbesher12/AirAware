import type { RiskLevel } from '../../types/sensors'

export default function RiskBadge({ level, compact = false }: { level: RiskLevel | null; compact?: boolean }) {
  const tone = level?.toLowerCase() ?? 'unavailable'
  return <span className={`airaware-risk-badge risk-${tone}${compact ? ' compact' : ''}`}>
    <i />{level ?? 'Unavailable'}
  </span>
}
