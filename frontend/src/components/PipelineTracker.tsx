import React from 'react'

interface PipelineTrackerProps {
  activeStage: number
}

interface StageDef {
  id: number
  label: string
  locked: boolean
}

const STAGES: StageDef[] = [
  { id: 1, label: 'CNN', locked: false },
  { id: 2, label: 'Refiner', locked: false },
  { id: 3, label: 'Diffusion', locked: true },
  { id: 4, label: 'Ensemble', locked: true },
  { id: 5, label: 'Classify', locked: true },
]

const CheckIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

const LockIcon = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
)

const ActiveDot = () => (
  <svg width="6" height="6" viewBox="0 0 6 6">
    <circle cx="3" cy="3" r="2" fill="currentColor" />
  </svg>
)

function getStageState(stageId: number, activeStage: number): 'completed' | 'active' | 'locked' | 'idle' {
  if (STAGES.find(s => s.id === stageId)?.locked && stageId > activeStage) return 'locked'
  if (stageId < activeStage) return 'completed'
  if (stageId === activeStage) return 'active'
  return 'idle'
}

export default function PipelineTracker({ activeStage }: PipelineTrackerProps) {
  const elements: React.JSX.Element[] = []
  STAGES.forEach((stage, i) => {
    const state = getStageState(stage.id, activeStage)
    elements.push(
      <div key={`s${stage.id}`} className="pipeline-stage">
        <div className={`pipeline-circle ${state}`}>
          {state === 'completed' && <CheckIcon />}
          {state === 'active' && <ActiveDot />}
          {state === 'locked' && <LockIcon />}
          {state === 'idle' && (
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--text-muted)', opacity: 0.3, display: 'block' }} />
          )}
        </div>
        <span className={`pipeline-label ${state}`}>{stage.label}</span>
      </div>
    )
    if (i < STAGES.length - 1) {
      elements.push(
        <div key={`l${stage.id}`} className={`pipeline-line ${stage.id < activeStage ? 'completed' : ''}`} />
      )
    }
  })

  return (
    <div className="pipeline-bar">
      <div className="pipeline-stages">
        {elements}
      </div>
    </div>
  )
}

PipelineTracker.displayName = 'PipelineTracker'
