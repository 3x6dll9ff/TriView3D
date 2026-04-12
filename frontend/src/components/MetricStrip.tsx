interface MetricDef {
  key: string
  label: string
  value: number | string
  unit?: string
}

interface MetricStripProps {
  metrics: MetricDef[]
  primaryKeys?: string[]
}

const KEY_METRICS = ['dice', 'reproj', 'assd', 'vol']

function getQualityClass(key: string, val: number | string): string {
  const n = typeof val === 'number' ? val : parseFloat(val as string)
  if (isNaN(n)) return ''
  if (['dice', 'iou', 'precision', 'recall', 'sim'].includes(key)) {
    if (n >= 0.85) return 'good'
    if (n >= 0.7) return 'ok'
    return 'bad'
  }
  if (key === 'reproj') {
    if (n <= 0.03) return 'good'
    if (n <= 0.06) return 'ok'
    return 'bad'
  }
  if (key === 'vol') {
    if (n <= 5) return 'good'
    if (n <= 15) return 'ok'
    return 'bad'
  }
  if (key === 'assd' || key === 'hd95') {
    if (n <= 1.0) return 'good'
    if (n <= 3.0) return 'ok'
    return 'bad'
  }
  return ''
}

function formatVal(v: number | string, unit?: string): string {
  if (v === undefined || v === null) return '—'
  const num = typeof v === 'number' ? v : parseFloat(v as string)
  if (isNaN(num)) return '—'
  return `${num.toFixed(3)}${unit ? ` ${unit}` : ''}`
}

export default function MetricStrip({ metrics, primaryKeys = KEY_METRICS }: MetricStripProps) {
  const primary = primaryKeys
    .map(k => metrics.find(m => m.key === k))
    .filter((m): m is MetricDef => m !== undefined)

  const secondary = metrics.filter(m => !primaryKeys.includes(m.key))

  if (primary.length === 0) return null

  return (
    <div className="metric-strip">
      {primary.map(m => (
        <div key={m.key} className={`metric-strip-item ${getQualityClass(m.key, m.value)}`}>
          <span className="metric-strip-label">{m.label}</span>
          <span className="metric-strip-value">{formatVal(m.value, m.unit)}</span>
          {secondary.length > 0 && (
            <div className="metric-strip-tooltip">
              {secondary.map(s => (
                <div key={s.key} className="metric-strip-tooltip-row">
                  <span className="metric-strip-tooltip-label">{s.label}</span>
                  <span className={`metric-strip-tooltip-value ${getQualityClass(s.key, s.value)}`}>
                    {formatVal(s.value, s.unit)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

MetricStrip.displayName = 'MetricStrip'
