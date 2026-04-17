import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useThree, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import Sidebar from './components/Sidebar'
import MetricStrip from './components/MetricStrip'
import PipelineTracker from './components/PipelineTracker'
import './index.css'

const sharedCamera = {
  position: new THREE.Vector3(0, 40, 50),
  target: new THREE.Vector3(0, 0, 0),
  quaternion: new THREE.Quaternion(),
}

function SyncedControls() {
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)

  useFrame(() => {
    if (!controlsRef.current) return
    camera.position.copy(sharedCamera.position)
    camera.quaternion.copy(sharedCamera.quaternion)
    controlsRef.current.target.copy(sharedCamera.target)
    controlsRef.current.update()
  })

  const handleChange = () => {
    if (!controlsRef.current) return
    sharedCamera.position.copy(camera.position)
    sharedCamera.quaternion.copy(camera.quaternion)
    sharedCamera.target.copy(controlsRef.current.target)
  }

  return (
    <OrbitControls
      ref={controlsRef}
      onChange={handleChange}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      enableDamping={true}
      dampingFactor={0.05}
      makeDefault
    />
  )
}

function GridFloor() {
  return (
    <group position={[0, -32, 0]}>
      <gridHelper args={[80, 20, '#1a1a2e', '#1a1a2e']} />
      <gridHelper args={[80, 4, '#2a2a3e', '#2a2a3e']} />
    </group>
  )
}

function ScaleBar() {
  return (
    <group position={[24, -30, 24]}>
      <mesh>
        <boxGeometry args={[20, 0.12, 0.12]} />
        <meshBasicMaterial color="#4fffff" transparent opacity={0.5} />
      </mesh>
      <mesh position={[-10, 0.4, 0]}>
        <boxGeometry args={[0.12, 0.8, 0.12]} />
        <meshBasicMaterial color="#4fffff" transparent opacity={0.5} />
      </mesh>
      <mesh position={[10, 0.4, 0]}>
        <boxGeometry args={[0.12, 0.8, 0.12]} />
        <meshBasicMaterial color="#4fffff" transparent opacity={0.5} />
      </mesh>
    </group>
  )
}

const circleTexture = (() => {
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 64
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.arc(32, 32, 30, 0, Math.PI * 2)
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  return new THREE.CanvasTexture(canvas)
})()

function CellMesh({ vertices, indices, color, baseOpacity, isOverlay, overlayOpacity }: {
  vertices: number[]
  indices: number[]
  color: string
  baseOpacity?: number
  isOverlay?: boolean
  overlayOpacity?: number
}) {
  const op = baseOpacity ?? 0.18
  const geometry = useMemo(() => {
    if (!vertices?.length || !indices?.length) return null

    let minV0 = Infinity, maxV0 = -Infinity
    for (let i = 0; i < vertices.length; i += 3) {
      if (vertices[i] < minV0) minV0 = vertices[i]
      if (vertices[i] > maxV0) maxV0 = vertices[i]
    }

    const flip = (minV0 - 32) < (maxV0 - 32)
    const verts = new Float32Array(vertices.length)
    for (let i = 0; i < vertices.length; i += 3) {
      verts[i] = vertices[i + 2] - 32
      verts[i + 1] = flip ? (32 - vertices[i]) : (vertices[i] - 32)
      verts[i + 2] = vertices[i + 1] - 32
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3))
    geo.setIndex(indices)
    geo.computeVertexNormals()
    return geo
  }, [vertices, indices])

  if (!geometry) return null

  if (isOverlay) {
    const ovOp = overlayOpacity ?? 0.6
    return (
      <group>
        <mesh geometry={geometry}>
          <meshStandardMaterial
            color={color}
            transparent
            opacity={ovOp * 0.5}
            side={THREE.DoubleSide}
            roughness={0.4}
            metalness={0.1}
            depthWrite={false}
          />
        </mesh>
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color={color}
            wireframe
            transparent
            opacity={ovOp * 0.9}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
        <points geometry={geometry}>
          <pointsMaterial
            size={0.4}
            color={color}
            transparent
            opacity={ovOp * 0.8}
            map={circleTexture}
            alphaMap={circleTexture}
            alphaTest={0.01}
            sizeAttenuation
            depthWrite={false}
          />
        </points>
      </group>
    )
  }

    return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color={color}
          transparent
          opacity={op}
          side={THREE.DoubleSide}
          roughness={0.6}
          metalness={0.05}
          depthWrite={true}
        />
      </mesh>
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color={color}
          wireframe
          transparent
          opacity={0.12}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <points geometry={geometry}>
        <pointsMaterial
          size={0.3}
          color={color}
          transparent
          opacity={0.9}
          map={circleTexture}
          alphaMap={circleTexture}
          alphaTest={0.01}
          sizeAttenuation
          depthWrite={false}
        />
      </points>
    </group>
  )
}

function Scene({ meshData, color, label, overlayMesh, overlayColor, overlayOpacity, dimBase }: {
  meshData: { vertices: number[]; indices: number[] } | null
  color: string
  label: string
  overlayMesh?: { vertices: number[]; indices: number[] } | null
  overlayColor?: string
  overlayOpacity?: number
  dimBase?: boolean
}) {
  return (
    <div className="scene-container">
      <div className="scene-label">{label}</div>
      <Canvas camera={{ position: [0, 40, 50], fov: 45 }} gl={{ antialias: true }}>
        <color attach="background" args={['#08080f']} />
        <ambientLight intensity={0.25} />
        <directionalLight position={[30, 50, 20]} intensity={1.2} color="#ffffff" />
        <directionalLight position={[-20, 10, -30]} intensity={0.4} color={color} />
        <pointLight position={[0, -20, 0]} intensity={0.5} color="#6366f1" />
        <GridFloor />
        <ScaleBar />
        {meshData && (
          <CellMesh vertices={meshData.vertices} indices={meshData.indices} color={color} baseOpacity={dimBase ? 0.08 : undefined} />
        )}
        {overlayMesh && overlayColor && (
          <CellMesh
            vertices={overlayMesh.vertices}
            indices={overlayMesh.indices}
            color={overlayColor}
            isOverlay
            overlayOpacity={overlayOpacity}
          />
        )}
        <SyncedControls />
      </Canvas>
    </div>
  )
}

interface CellInfo { filename: string; score: string; type: string }
interface MetricDef { key: string; label: string; value: number | string; unit?: string }
type PreviewMap = Record<string, string>

interface PredictionEntry {
  id: number
  timestamp: string
  filename: string
  cellType: string
  model: string
  dice: number
  iou: number
  precision: number
  recall: number
  assd: number | null
  hd95: number | null
  volumeDiffPct: number | null
  reprojectionL1: number | null
}

const API = import.meta.env.VITE_API_BASE_URL || ''

function App() {
  const [cells, setCells] = useState<CellInfo[]>([])
  const [selectedCell, setSelectedCell] = useState('')
  const [cnnData, setCnnData] = useState<any>(null)
  const [vaeData, setVaeData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [vaeAvailable, setVaeAvailable] = useState(false)
  const [tab, setTab] = useState<'predict' | 'metrics'>('predict')
  const [metricsHistory, setMetricsHistory] = useState<any[]>([])
  const [vaeHistory, setVaeHistory] = useState<any[]>([])
  const [vaeRaw, setVaeRaw] = useState<any>(null)
  const [preview, setPreview] = useState<PreviewMap | null>(null)
  const [predLog, setPredLog] = useState<PredictionEntry[]>([])
  const [metricsSub, setMetricsSub] = useState<'generations' | 'training'>('generations')

  useEffect(() => {
    axios.get(`${API}/api/cells`).then(res => {
      setCells(res.data.cells)
      if (res.data.cells.length > 0) setSelectedCell(res.data.cells[0].filename)
    }).catch(console.error)

    axios.get(`${API}/api/metrics`).then(res => {
      const d = res.data
      if (d?.train_loss) {
        const lossKey = d?.test_loss ? 'test_loss' : 'val_loss'
        const diceKey = d?.test_dice ? 'test_dice' : 'val_dice'
        const iouKey = d?.test_iou ? 'test_iou' : 'val_iou'
        setMetricsHistory(
          d.train_loss.map((_: number, i: number) => ({
            epoch: i + 1,
            train_loss: d.train_loss[i],
            train_bce: d.train_bce?.[i],
            train_dice_loss: d.train_dice_loss?.[i],
            train_projection: d.train_projection?.[i],
            train_surface: d.train_surface?.[i],
            val_loss: d[lossKey]?.[i],
            val_projection: d.val_projection?.[i],
            val_dice: d[diceKey]?.[i],
            val_iou: d[iouKey]?.[i],
            val_hard_dice: d.val_hard_dice?.[i],
            val_hard_iou: d.val_hard_iou?.[i],
          }))
        )
      }
    }).catch(console.error)

    axios.get(`${API}/api/metrics-vae`).then(res => {
      const d = res.data
      if (d?.train_loss) {
        setVaeRaw(d)
        setVaeHistory(
          d.train_loss.map((_: number, i: number) => ({
            epoch: i + 1,
            train_loss: d.train_loss[i],
            train_recon: d.train_recon?.[i],
            train_kl: d.train_kl?.[i],
            train_projection: d.train_projection?.[i],
            train_surface: d.train_surface?.[i],
            test_loss: d.test_loss?.[i],
            test_projection: d.test_projection?.[i],
            test_dice: d.test_dice?.[i],
            test_iou: d.test_iou?.[i],
          }))
        )
      }
    }).catch(console.error)

    axios.get(`${API}/api/status`).then(res => {
      setVaeAvailable(res.data.vae_loaded)
    }).catch(() => setVaeAvailable(false))
  }, [])

  useEffect(() => {
    if (!selectedCell) return
    axios.get(`${API}/api/preview/${selectedCell}`)
      .then(res => setPreview(res.data))
      .catch(console.error)
  }, [selectedCell])

  const handlePredict = useCallback(async () => {
    if (!selectedCell) return
    setLoading(true)
    setCnnData(null)
    setVaeData(null)
    try {
      const cnnPromise = axios.post(`${API}/api/predict/${selectedCell}`)
      const vaePromise = vaeAvailable
        ? axios.post(`${API}/api/predict-vae/${selectedCell}`).catch(() => null)
        : Promise.resolve(null)

      const [cnnRes, vaeRes] = await Promise.all([cnnPromise, vaePromise])
      setCnnData(cnnRes.data)
      if (vaeRes) setVaeData(vaeRes.data)

      const cellInfo = cells.find(c => c.filename === selectedCell)
      const now = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

      if (cnnRes.data) {
        const m = cnnRes.data.metrics ?? {}
        setPredLog(prev => [...prev, {
          id: Date.now(),
          timestamp: now,
          filename: selectedCell,
          cellType: cellInfo?.type ?? '',
          model: 'CNN+Refiner',
          dice: m.dice ?? cnnRes.data.dice ?? 0,
          iou: m.iou ?? 0,
          precision: m.precision ?? 0,
          recall: m.recall ?? 0,
          assd: m.surface_assd ?? null,
          hd95: m.surface_hd95 ?? null,
          volumeDiffPct: m.volume_diff_pct ?? null,
          reprojectionL1: m.reprojection_l1 ?? null,
        }])
      }
      if (vaeRes?.data) {
        const m = vaeRes.data.metrics ?? {}
        setPredLog(prev => [...prev, {
          id: Date.now() + 1,
          timestamp: now,
          filename: selectedCell,
          cellType: cellInfo?.type ?? '',
          model: 'CVAE',
          dice: m.dice ?? vaeRes.data.dice ?? 0,
          iou: m.iou ?? 0,
          precision: m.precision ?? 0,
          recall: m.recall ?? 0,
          assd: m.surface_assd ?? null,
          hd95: m.surface_hd95 ?? null,
          volumeDiffPct: m.volume_diff_pct ?? null,
          reprojectionL1: m.reprojection_l1 ?? null,
        }])
      }
    } catch {
      alert('Backend error. Is FastAPI running on :8000?')
    } finally {
      setLoading(false)
    }
  }, [selectedCell, vaeAvailable, cells])

  const buildMetrics = (data: any): MetricDef[] => data ? [
    { key: 'dice', label: 'Dice', value: data.metrics?.dice ?? data.dice },
    { key: 'iou', label: 'IoU', value: data.metrics?.iou },
    { key: 'precision', label: 'Precision', value: data.metrics?.precision },
    { key: 'recall', label: 'Recall', value: data.metrics?.recall },
    { key: 'reproj', label: 'Reproj L1', value: data.metrics?.reprojection_l1 },
    { key: 'assd', label: 'ASSD', value: data.metrics?.surface_assd, unit: 'vox' },
    { key: 'hd95', label: 'HD95', value: data.metrics?.surface_hd95, unit: 'vox' },
    { key: 'sim', label: 'Surf Sim', value: data.metrics?.surface_similarity },
    { key: 'vol', label: 'Vol Diff', value: data.metrics?.volume_diff_pct, unit: '%' },
  ] : []

  const activeStage = useMemo(() => {
    if (loading && !cnnData) return 1
    if (cnnData && loading) return 2
    if (cnnData) return 3
    return 0
  }, [loading, cnnData])

  const hasResults = cnnData || vaeData
  const [overlay, setOverlay] = useState(false)
  const [overlayOpacity, setOverlayOpacity] = useState(0.6)

  return (
    <div className="app">
      <Sidebar tab={tab} onTabChange={setTab} vaeAvailable={vaeAvailable} />

      <div className="main-area">
        <header className="header">
          <div className="header-left">
            <div className="header-status">
              <div className={`status-dot ${cnnData || vaeData ? 'online' : ''}`} />
              <span className="header-info">
                {cells.length} samples · {vaeAvailable ? 'VAE + CNN' : 'CNN only'}
              </span>
            </div>
          </div>
        </header>

        {tab === 'predict' && (
          <main className="content" style={{ paddingTop: 0 }}>
            <section className="top-block">
              <div className="top-block-left">
                <div className="select-wrapper">
                  <label className="field-label">Cell Sample</label>
                  <select
                    className="select"
                    value={selectedCell}
                    onChange={e => setSelectedCell(e.target.value)}
                  >
                    {cells.map(c => (
                      <option key={c.filename} value={c.filename}>
                        {c.type} · score {c.score}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  className="btn-predict"
                  onClick={handlePredict}
                  disabled={loading}
                >
                  {loading ? 'Processing...' : 'Generate'}
                </button>
              </div>

              <div className="top-block-mid">
                {preview ? (
                  ['top', 'bottom', 'side', 'front'].filter(view => preview[view as keyof typeof preview]).map(view => (
                    <div key={view} className="projection-card">
                      <span className="projection-label">{view}</span>
                      <img
                        src={preview[view as keyof typeof preview]}
                        alt={`${view} projection`}
                        className="projection-img"
                      />
                    </div>
                  ))
                ) : (
                  <div className="top-block-empty">Select a sample</div>
                )}
              </div>

              <div className="top-block-pipeline">
                <PipelineTracker activeStage={activeStage} />
                {hasResults && (
                  <div className="overlay-controls">
                    <button
                      className={`overlay-btn ${overlay ? 'overlay-btn-active' : ''}`}
                      onClick={() => setOverlay(!overlay)}
                    >
                      Overlay
                    </button>
                    {overlay && (
                      <div className="overlay-slider-wrap">
                        <span className="overlay-slider-label">GT</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={overlayOpacity}
                          onChange={e => setOverlayOpacity(parseFloat(e.target.value))}
                          className="overlay-slider"
                        />
                        <span className="overlay-slider-val">{Math.round(overlayOpacity * 100)}%</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>

            {(loading || hasResults) && (
              <section className={`viewports ${vaeData ? 'viewports-3' : ''}`}>
                  <div className="viewport">
                    {cnnData && <MetricStrip metrics={buildMetrics(cnnData)} />}
                    {loading && !cnnData && (
                      <div className="viewport-loading">
                        <div className="spinner" />
                      </div>
                    )}
                    <Scene
                      meshData={cnnData?.pred}
                      color="#4fffff"
                      label="CNN Prediction"
                      overlayMesh={overlay ? cnnData?.gt : undefined}
                      overlayColor="#ef4444"
                      overlayOpacity={overlayOpacity}
                      dimBase={overlay}
                    />
                  </div>
                  {vaeData && (
                    <div className="viewport">
                      <MetricStrip metrics={buildMetrics(vaeData)} />
                      <Scene
                        meshData={vaeData?.pred}
                        color="#a0c4ff"
                        label="VAE Generation"
                        overlayMesh={overlay ? cnnData?.gt : undefined}
                        overlayColor="#ef4444"
                        overlayOpacity={overlayOpacity}
                        dimBase={overlay}
                      />
                    </div>
                  )}
                  <div className="viewport">
                    {cnnData && <MetricStrip metrics={[
                      { key: 'gt_info', label: 'Ground Truth', value: 'reference' },
                      { key: 'gt_dice', label: 'Ref Dice', value: cnnData?.dice ?? '—' },
                      { key: 'gt_reproj', label: 'Reproj L1', value: cnnData?.metrics?.reprojection_l1 ?? '—' },
                      { key: 'gt_vol', label: 'Vol Diff', value: cnnData?.metrics?.volume_diff_pct ?? '—', unit: '%' },
                    ]} />}
                    <Scene
                      meshData={cnnData?.gt}
                      color="#a5d8ff"
                      label="Ground Truth"
                      overlayMesh={overlay ? cnnData?.pred : undefined}
                      overlayColor="#4fffff"
                      overlayOpacity={overlayOpacity}
                      dimBase={overlay}
                    />
                  </div>
              </section>
            )}

            {!hasResults && !loading && (
              <div className="empty-state">
                Select a cell sample and click <strong>Generate</strong> to begin
              </div>
            )}
          </main>
        )}

        {tab === 'metrics' && (
          <main className="content">
            <div className="metrics-model-selector">
              <button
                className={`metrics-model-btn ${metricsSub === 'generations' ? 'metrics-model-btn-active' : ''}`}
                onClick={() => setMetricsSub('generations')}
              >
                Generations Log
              </button>
              <button
                className={`metrics-model-btn ${metricsSub === 'training' ? 'metrics-model-btn-active' : ''}`}
                onClick={() => setMetricsSub('training')}
              >
                Training Curves
              </button>
            </div>

            {metricsSub === 'generations' && (
              <>
            {predLog.length > 0 && (
              <>
            <section className="chart-section">
              <h2 className="chart-title">Dice Score per Generation</h2>
              <p className="chart-subtitle">Reconstruction quality across all generated cells</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={predLog.map((e, i) => ({
                    idx: i + 1,
                    name: e.filename.replace('.npy', ''),
                    dice: e.dice,
                    model: e.model,
                  }))} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="idx" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" label={{ value: '#', position: 'insideBottomRight', fontSize: 9, fill: 'rgba(255,255,255,0.2)' }} />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" domain={[0.5, 1.0]} />
                    <Tooltip
                      contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                    />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Dice" dataKey="dice" stroke="#4fffff" strokeWidth={2} dot={{ r: 4, fill: '#4fffff', stroke: '#4fffff' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="chart-section">
              <h2 className="chart-title">Summary Statistics</h2>
              <p className="chart-subtitle">Aggregated metrics across {predLog.length} generation{predLog.length !== 1 ? 's' : ''}</p>
              <div className="summary-grid">
                <div className="summary-header">
                  <span className="summary-header-label">Metric</span>
                  <span className="summary-header-val" style={{ color: '#4fffff' }}>CNN+Refiner</span>
                  <span className="summary-header-val" style={{ color: '#a0c4ff' }}>CVAE</span>
                </div>
                {(() => {
                  const cnnLog = predLog.filter(e => e.model === 'CNN+Refiner')
                  const vaeLog = predLog.filter(e => e.model === 'CVAE')
                  const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0
                  const mn = (arr: number[]) => arr.length ? Math.min(...arr) : 0
                  const mx = (arr: number[]) => arr.length ? Math.max(...arr) : 0
                  const rows = [
                    { label: 'Generations', cnn: cnnLog.length, vae: vaeLog.length },
                    { label: 'Avg Dice', cnn: avg(cnnLog.map(e => e.dice)).toFixed(4), vae: avg(vaeLog.map(e => e.dice)).toFixed(4) },
                    { label: 'Min Dice', cnn: mn(cnnLog.map(e => e.dice)).toFixed(4), vae: mn(vaeLog.map(e => e.dice)).toFixed(4) },
                    { label: 'Max Dice', cnn: mx(cnnLog.map(e => e.dice)).toFixed(4), vae: mx(vaeLog.map(e => e.dice)).toFixed(4) },
                    { label: 'Avg IoU', cnn: avg(cnnLog.map(e => e.iou)).toFixed(4), vae: avg(vaeLog.map(e => e.iou)).toFixed(4) },
                    { label: 'Avg ASSD', cnn: avg(cnnLog.map(e => e.assd ?? 0)).toFixed(2), vae: avg(vaeLog.map(e => e.assd ?? 0)).toFixed(2) },
                    { label: 'Avg HD95', cnn: avg(cnnLog.map(e => e.hd95 ?? 0)).toFixed(2), vae: avg(vaeLog.map(e => e.hd95 ?? 0)).toFixed(2) },
                  ]
                  return rows.map(r => (
                    <div key={r.label} className="summary-row">
                      <span className="summary-label">{r.label}</span>
                      <span className="summary-val" style={{ color: cnnLog.length ? '#4fffff' : 'var(--text-muted)' }}>{String(r.cnn)}</span>
                      <span className="summary-val" style={{ color: vaeLog.length ? '#a0c4ff' : 'var(--text-muted)' }}>{String(r.vae)}</span>
                    </div>
                  ))
                })()}
              </div>
            </section>
              </>
            )}

            <section className="chart-section">
              <h2 className="chart-title">Generation History</h2>
              <p className="chart-subtitle">{predLog.length ? `${predLog.length} generation${predLog.length !== 1 ? 's' : ''} recorded` : 'Generate cells in the Predictor tab to see results here'}</p>
              {predLog.length > 0 ? (
                <div className="gen-table-wrap">
                  <table className="gen-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Time</th>
                        <th>Cell</th>
                        <th>Type</th>
                        <th>Model</th>
                        <th>Dice</th>
                        <th>IoU</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>ASSD</th>
                        <th>HD95</th>
                        <th>Vol Diff</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predLog.map((e, i) => (
                        <tr key={e.id}>
                          <td>{i + 1}</td>
                          <td className="mono-sm">{e.timestamp}</td>
                          <td className="mono-sm" title={e.filename}>{e.filename.replace('.npy', '').substring(0, 20)}</td>
                          <td>{e.cellType}</td>
                          <td><span className={`model-tag ${e.model === 'CVAE' ? 'model-tag-vae' : 'model-tag-cnn'}`}>{e.model}</span></td>
                          <td className={`dice-val ${e.dice >= 0.95 ? 'dice-good' : e.dice >= 0.85 ? 'dice-ok' : 'dice-bad'}`}>{e.dice.toFixed(4)}</td>
                          <td>{e.iou.toFixed(4)}</td>
                          <td>{e.precision.toFixed(4)}</td>
                          <td>{e.recall.toFixed(4)}</td>
                          <td>{e.assd !== null ? e.assd.toFixed(2) : '—'}</td>
                          <td>{e.hd95 !== null ? e.hd95.toFixed(2) : '—'}</td>
                          <td>{e.volumeDiffPct !== null ? `${e.volumeDiffPct.toFixed(1)}%` : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state" style={{ minHeight: 120 }}>No generations yet</div>
              )}
            </section>
              </>
            )}

            {metricsSub === 'training' && (
              <>
            {metricsHistory.length > 0 && (
            <section className="chart-section">
              <h2 className="chart-title">CNN Training — Loss Convergence</h2>
              <p className="chart-subtitle">Composite BCE + Dice + Projection + Surface loss</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <Tooltip contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Train Loss" dataKey="train_loss" stroke="#4fffff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Val Loss" dataKey="val_loss" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            )}

            {metricsHistory.length > 0 && (
            <section className="chart-section">
              <h2 className="chart-title">CNN Training — Loss Components</h2>
              <p className="chart-subtitle">Breakdown: BCE, Dice, Projection, Surface</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <Tooltip contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="BCE" dataKey="train_bce" stroke="#4fffff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Dice" dataKey="train_dice_loss" stroke="#22c55e" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Projection" dataKey="train_projection" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Surface" dataKey="train_surface" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            )}

            {metricsHistory.length > 0 && (
            <section className="chart-section">
              <h2 className="chart-title">CNN Training — Reconstruction Quality</h2>
              <p className="chart-subtitle">Dice, Hard Dice, IoU on validation set</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" domain={[0.5, 1.0]} />
                    <Tooltip contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Dice" dataKey="val_dice" stroke="#4fffff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Hard Dice" dataKey="val_hard_dice" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="IoU" dataKey="val_iou" stroke="#22c55e" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            )}

            {vaeHistory.length > 0 && (
            <section className="chart-section">
              <h2 className="chart-title">CVAE Training — Loss Convergence</h2>
              <p className="chart-subtitle">Reconstruction + KL + Projection + Surface</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={vaeHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <Tooltip contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Train Loss" dataKey="train_loss" stroke="#a0c4ff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Test Loss" dataKey="test_loss" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            )}

            {vaeHistory.length > 0 && (
            <section className="chart-section">
              <h2 className="chart-title">CVAE Training — KL & Reconstruction</h2>
              <p className="chart-subtitle">KL divergence (latent regularisation) and reconstruction loss</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={vaeHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <Tooltip contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Reconstruction" dataKey="train_recon" stroke="#4fffff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="KL Divergence" dataKey="train_kl" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            )}

            {vaeHistory.length > 0 && (
            <section className="chart-section">
              <h2 className="chart-title">CVAE Training — Reconstruction Quality</h2>
              <p className="chart-subtitle">Best-of-K={vaeRaw?.config?.eval_samples_k ?? 8} sampling</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={vaeHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" domain={[0.5, 1.0]} />
                    <Tooltip contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Dice" dataKey="test_dice" stroke="#a0c4ff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="IoU" dataKey="test_iou" stroke="#22c55e" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            )}

            {metricsHistory.length === 0 && vaeHistory.length === 0 && (
              <div className="empty-state">No training history available</div>
            )}
              </>
            )}
          </main>
        )}
      </div>
    </div>
  )
}

export default App
