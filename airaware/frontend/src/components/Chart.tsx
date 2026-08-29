import { Component, type ErrorInfo, type ReactNode } from 'react'
import PlotModule from 'react-plotly.js'

function unwrapDefault(moduleValue: any): React.ComponentType<any> {
  let value = moduleValue
  while (value && typeof value === 'object' && 'default' in value) value = value.default
  return value as React.ComponentType<any>
}

const Plot = unwrapDefault(PlotModule)

class ChartBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('AirAware chart rendering failed', error, info)
  }

  render() {
    if (this.state.failed) {
      return <div className="chart-error"><strong>Chart unavailable</strong><span>The rest of the dashboard remains available.</span></div>
    }
    return this.props.children
  }
}
type Props = {
  data: any[]
  layout?: any
  className?: string
}

export default function Chart({ data, layout, className = '' }: Props) {
  return (
    <ChartBoundary>
      <Plot
        data={data}
        layout={{
          autosize: true,
          height: 340,
          margin: { l: 54, r: 24, t: 30, b: 48 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'Inter, system-ui, sans-serif', color: '#60706a', size: 11 },
          hoverlabel: { bgcolor: '#102820', bordercolor: '#2dd4a0', font: { color: '#f4faf7' } },
          hovermode: 'x unified',
          xaxis: { gridcolor: '#e5ebe8', zerolinecolor: '#d7e0dc', ...layout?.xaxis },
          yaxis: { gridcolor: '#e5ebe8', zerolinecolor: '#d7e0dc', ...layout?.yaxis },
          legend: { orientation: 'h', y: 1.1, x: 0 },
          ...layout,
        }}
        config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
        useResizeHandler
        className={`plot ${className}`}
        style={{ width: '100%', height: '100%' }}
      />
    </ChartBoundary>
  )
}
