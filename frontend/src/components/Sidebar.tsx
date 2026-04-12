import { useState, useEffect } from 'react'

interface SidebarProps {
  tab: 'predict' | 'metrics'
  onTabChange: (tab: 'predict' | 'metrics') => void
  vaeAvailable: boolean
}

const CubeIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
    <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
    <line x1="12" y1="22.08" x2="12" y2="12" />
  </svg>
)

const ChartIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
)

const ArrowLeftIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
)

const MenuIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="3" y1="6" x2="21" y2="6" />
    <line x1="3" y1="12" x2="21" y2="12" />
    <line x1="3" y1="18" x2="21" y2="18" />
  </svg>
)

export default function Sidebar({ tab, onTabChange, vaeAvailable }: SidebarProps) {
  const [expanded, setExpanded] = useState(() => {
    const stored = localStorage.getItem('sidebar-expanded')
    return stored !== null ? stored === 'true' : true
  })

  useEffect(() => {
    localStorage.setItem('sidebar-expanded', String(expanded))
  }, [expanded])

  return (
    <nav className={`sidebar ${expanded ? '' : 'collapsed'}`}>
      <div className="sidebar-brand">
        <span style={{ color: 'var(--accent)', fontSize: '16px', fontWeight: 700, flexShrink: 0, letterSpacing: '-0.03em' }}>T3</span>
        <div className="sidebar-brand-text">
          <div className="sidebar-brand-name">TriView<span>3D</span></div>
          <div className="sidebar-brand-sub">Reconstruction</div>
        </div>
      </div>

      <button
        className="sidebar-toggle"
        onClick={() => setExpanded(!expanded)}
        title={expanded ? 'Collapse' : 'Expand'}
      >
        {expanded ? <ArrowLeftIcon /> : <MenuIcon />}
      </button>

      <div className="sidebar-nav">
        <button
          className={`sidebar-item ${tab === 'predict' ? 'active' : ''}`}
          onClick={() => onTabChange('predict')}
          title="Predictor"
        >
          <span className="sidebar-item-icon"><CubeIcon /></span>
          <span className="sidebar-item-label">Predictor</span>
        </button>

        <button
          className={`sidebar-item ${tab === 'metrics' ? 'active' : ''}`}
          onClick={() => onTabChange('metrics')}
          title="Metrics"
        >
          <span className="sidebar-item-icon"><ChartIcon /></span>
          <span className="sidebar-item-label">Training Metrics</span>
        </button>
      </div>

      {expanded && vaeAvailable && (
        <div style={{ padding: '8px 16px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.1em', opacity: 0.6 }}>
            VAE Online
          </div>
        </div>
      )}
    </nav>
  )
}

Sidebar.displayName = 'Sidebar'
