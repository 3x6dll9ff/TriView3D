import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
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

const circleTexture = (() => {
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 64
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.arc(32, 32, 30, 0, Math.PI * 2)
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  const tex = new THREE.CanvasTexture(canvas)
  return tex
})()

function CellMesh({ vertices, indices, color }: {
  vertices: number[]
  indices: number[]
  color: string
}) {
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

  return (
    <group>
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.06}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color={color}
          wireframe
          transparent
          opacity={0.08}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <points geometry={geometry}>
        <pointsMaterial
          size={0.25}
          color={color}
          transparent
          opacity={0.95}
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

function Scene({ meshData, color, label }: {
  meshData: { vertices: number[]; indices: number[] } | null
  color: string
  label: string
}) {
  return (
    <div className="scene-container">
      <div className="scene-label">{label}</div>
      <Canvas camera={{ position: [0, 40, 50], fov: 45 }}>
        <color attach="background" args={['#08080f']} />
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1.2} color={color} />
        <pointLight position={[-10, -10, -10]} intensity={0.8} color="#6366f1" />
        <Environment preset="city" />
        {meshData && (
          <CellMesh vertices={meshData.vertices} indices={meshData.indices} color={color} />
        )}
        <SyncedControls />
      </Canvas>
    </div>
  )
}

interface CellInfo { filename: string; score: string; type: string }
interface MetricDef { key: string; label: string; value: number | string; unit?: string }
type PreviewMap = Record<string, string>

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
  const [preview, setPreview] = useState<PreviewMap | null>(null)

  useEffect(() => {
    axios.get(`${API}/api/cells`).then(res => {
      setCells(res.data.cells)
      if (res.data.cells.length > 0) setSelectedCell(res.data.cells[0].filename)
    }).catch(console.error)

    axios.get(`${API}/api/metrics`).then(res => {
      const lossKey = res.data?.test_loss ? 'test_loss' : 'val_loss'
      const diceKey = res.data?.test_dice ? 'test_dice' : 'val_dice'
      const iouKey = res.data?.test_iou ? 'test_iou' : 'val_iou'
      if (res.data?.train_loss && res.data?.[lossKey]) {
        setMetricsHistory(
          res.data.train_loss.map((_: number, i: number) => ({
            epoch: i + 1,
            train_loss: res.data.train_loss[i],
            test_loss: res.data[lossKey][i],
            test_dice: res.data[diceKey]?.[i],
            test_iou: res.data[iouKey]?.[i],
            test_hard_dice: res.data.val_hard_dice?.[i],
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
    } catch {
      alert('Backend error. Is FastAPI running on :8000?')
    } finally {
      setLoading(false)
    }
  }, [selectedCell, vaeAvailable])

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
                  />
                </div>
                {vaeData && (
                  <div className="viewport">
                    <MetricStrip metrics={buildMetrics(vaeData)} />
                    <Scene
                      meshData={vaeData?.pred}
                      color="#a0c4ff"
                      label="VAE Generation"
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
                  />
                </div>
              </section>
            )}

            {!hasResults && !loading && (
              <div className="empty-state">
                Select a cell sample and click <strong>Predict 3D Shape</strong> to begin
              </div>
            )}
          </main>
        )}

        {tab === 'metrics' && (
          <main className="content">
            <section className="chart-section">
              <h2 className="chart-title">Loss Convergence</h2>
              <p className="chart-subtitle">BCE + Dice hybrid loss, 50 epochs, Adam lr=1e-3</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <Tooltip
                      contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                    />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Train Loss" dataKey="train_loss" stroke="#4fffff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Test Loss" dataKey="test_loss" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="chart-section">
              <h2 className="chart-title">Reconstruction Quality</h2>
              <p className="chart-subtitle">Dice score and IoU on test set</p>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory} margin={{ left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" />
                    <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} fontFamily="JetBrains Mono" domain={[0.5, 1.0]} />
                    <Tooltip
                      contentStyle={{ background: '#18181f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                    />
                    <Legend iconType="circle" />
                    <Line type="monotone" name="Dice Score" dataKey="test_dice" stroke="#4fffff" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="Hard Dice" dataKey="test_hard_dice" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" name="IoU" dataKey="test_iou" stroke="#22c55e" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </main>
        )}
      </div>
    </div>
  )
}

export default App
