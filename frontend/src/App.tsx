import { useState, useMemo, useEffect, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars, Environment } from '@react-three/drei'
import { useThree, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

// Глобальные переменные для идеальной синхронизации двух независимых Canvas
const globalCameraPosition = new THREE.Vector3(0, 80, 100)
const globalCameraTarget = new THREE.Vector3(0, 0, 0)
const globalCameraRotation = new THREE.Quaternion()

function SyncedControls() {
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)

  // Каждый кадр мы читаем и применяем глобальную позицию
  useFrame(() => {
    if (!controlsRef.current) return
    // Если пользователь не вращает ЭТУ конкретную камеру, она должна принять глобальную позицию
    camera.position.copy(globalCameraPosition)
    camera.quaternion.copy(globalCameraRotation)
    controlsRef.current.target.copy(globalCameraTarget)
    controlsRef.current.update()
  })

  const handleChange = () => {
    if (!controlsRef.current) return
    // Когда пользователь крутит ЭТУ камеру, она записывает свои координаты в глобальный стейт
    globalCameraPosition.copy(camera.position)
    globalCameraRotation.copy(camera.quaternion)
    globalCameraTarget.copy(controlsRef.current.target)
  }

  return (
    <OrbitControls 
      ref={controlsRef} 
      onChange={handleChange}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      makeDefault 
    />
  )
}

import './index.css' // We will add glassmorphism styles here

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

function CellMesh({ vertices, indices, position, color }: any) {
  const geometry = useMemo(() => {
    if (!vertices || !indices || vertices.length === 0) return null
    
    // marching_cubes возвращает координаты в numpy-порядке [v0, v1, v2] ~ [Z, Y, X].
    // Фиксируем ориентацию так, чтобы ось top/bottom (v0) всегда была вертикальной (world Y):
    // top (малый v0) -> вверх, bottom (большой v0) -> вниз.
    let minV0 = Number.POSITIVE_INFINITY
    let maxV0 = Number.NEGATIVE_INFINITY
    for (let i = 0; i < vertices.length; i += 3) {
      const v0 = vertices[i]
      if (v0 < minV0) minV0 = v0
      if (v0 > maxV0) maxV0 = v0
    }

    // Если по текущему маппингу top оказался ниже bottom, инвертируем вертикальную ось.
    const shouldFlipVertical = (minV0 - 32) < (maxV0 - 32)

    const newVertices = new Float32Array(vertices.length)
    for (let i = 0; i < vertices.length; i += 3) {
      const v0 = vertices[i]      // Z in numpy (top/bottom axis)
      const v1 = vertices[i + 1]  // Y in numpy
      const v2 = vertices[i + 2]  // X in numpy

      newVertices[i] = v2 - 32
      newVertices[i + 1] = shouldFlipVertical ? (32 - v0) : (v0 - 32)
      newVertices[i + 2] = v1 - 32
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(newVertices, 3))
    geo.setIndex(indices)
    geo.computeVertexNormals()
    return geo
  }, [vertices, indices])

  if (!geometry) return null

  return (
    <group position={position}>
      {/* СВЯЗИ (Космическая паутина) - полупрозрачный каркас */}
      <mesh geometry={geometry}>
        <meshBasicMaterial 
          color={color} 
          wireframe={true} 
          transparent 
          opacity={0.1} 
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      
      {/* УЗЛЫ (Звезды) - светящиеся точки на вершинах графа */}
      <points geometry={geometry}>
        <pointsMaterial 
          size={0.6} 
          color={color}
          transparent 
          opacity={0.8} 
          blending={THREE.AdditiveBlending}
          sizeAttenuation={true}
          depthWrite={false}
        />
      </points>
    </group>
  )
}

function App() {
  const [cells, setCells] = useState<any[]>([])
  const [selectedCell, setSelectedCell] = useState('')
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'main'|'metrics'>('main')
  const [metricsHistory, setMetricsHistory] = useState<any[]>([])
  const [previewImages, setPreviewImages] = useState<{top:string, bottom:string, side:string} | null>(null)

  useEffect(() => {
    // Fetch available cells
    axios.get(`${API_BASE_URL}/api/cells`).then(res => {
      setCells(res.data.cells)
      if (res.data.cells.length > 0) setSelectedCell(res.data.cells[0].filename)
    }).catch(console.error)

    // Fetch metrics history
    axios.get(`${API_BASE_URL}/api/metrics`).then(res => {
      if (res.data && res.data.train_loss) {
        // Transform { train_loss: [], test_loss: [], ... } to array of objects for Recharts
        const len = res.data.train_loss.length;
        const formatted = Array.from({length: len}).map((_, i) => ({
          epoch: i + 1,
          train_loss: res.data.train_loss[i],
          test_loss: res.data.test_loss[i],
          test_dice: res.data.test_dice[i],
          test_iou: res.data.test_iou[i],
        }))
        setMetricsHistory(formatted)
      }
    }).catch(console.error)
  }, [])

  // Постоянно подгружать превью проекций при смене выбранной молекулы
  useEffect(() => {
    if (!selectedCell) return
    axios.get(`${API_BASE_URL}/api/preview/${selectedCell}`)
      .then(res => setPreviewImages(res.data))
      .catch(console.error)
  }, [selectedCell])

  const handlePredict = async () => {
    if (!selectedCell) return
    setLoading(true)
    try {
      const res = await axios.post(`${API_BASE_URL}/api/predict/${selectedCell}`)
      setData(res.data)
    } catch (e) {
      console.error(e)
      alert("Error talking to AI backend. Is FastAPI running on port 8000?")
    } finally {
      setLoading(false)
    }
  }

  const metricCards = data ? [
    {
      key: 'dice',
      label: 'VOXEL DICE',
      value: data.metrics?.dice ?? data.dice ?? '-',
      color: 'text-green-400',
      tooltip: 'Dice: пересечение объёмов предсказания и эталона. 1.0 = идеально, 0 = нет пересечения.',
    },
    {
      key: 'iou',
      label: 'IOU',
      value: data.metrics?.iou ?? '-',
      color: 'text-cyan-300',
      tooltip: 'IoU (Intersection over Union): доля пересечения от объединения объёмов. Более строгая, чем Dice.',
    },
    {
      key: 'precision',
      label: 'PRECISION',
      value: data.metrics?.precision ?? '-',
      color: 'text-sky-300',
      tooltip: 'Precision: какая доля предсказанных вокселей действительно относится к объекту.',
    },
    {
      key: 'recall',
      label: 'RECALL',
      value: data.metrics?.recall ?? '-',
      color: 'text-emerald-300',
      tooltip: 'Recall: какую долю вокселей эталонного объекта модель смогла восстановить.',
    },
    {
      key: 'surface_assd',
      label: 'SURFACE ASSD',
      value: data.metrics?.surface_assd ?? '-',
      color: 'text-violet-300',
      tooltip: 'ASSD: среднее расстояние между поверхностями (в вокселях). Чем меньше, тем лучше.',
    },
    {
      key: 'surface_hd95',
      label: 'SURFACE HD95',
      value: data.metrics?.surface_hd95 ?? '-',
      color: 'text-fuchsia-300',
      tooltip: 'HD95: 95-й перцентиль поверхностной ошибки (в вокселях). Показывает большие локальные расхождения.',
    },
    {
      key: 'surface_similarity',
      label: 'SURFACE SIM',
      value: data.metrics?.surface_similarity ?? '-',
      color: 'text-purple-300',
      tooltip: 'Surface Similarity: нормированная оценка из ASSD (1/(1+ASSD)). Ближе к 1 — ближе поверхности.',
    },
    {
      key: 'volume_diff_pct',
      label: 'VOL DIFF %',
      value: data.metrics?.volume_diff_pct ?? '-',
      color: 'text-amber-300',
      tooltip: 'Vol Diff %: относительная разница объёма предсказания и эталона. Чем ближе к 0%, тем лучше.',
    },
  ] : []

  return (
    <div className="w-full h-screen bg-black text-white flex overflow-hidden font-sans">
      {/* SIDEBAR (Pure Navigation) */}
      <div className="w-64 h-full backdrop-blur-md bg-white/5 border-r border-white/10 p-6 flex flex-col z-10 shadow-2xl">
        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500 mb-2">
          3D Cell AI
        </h1>
        <p className="text-xs text-gray-400 mb-12 uppercase tracking-widest">Dashboard</p>
        
        {/* TAB NAVIGATION */}
        <div className="flex flex-col gap-3">
          <button 
            onClick={() => setActiveTab('main')}
            className={`w-full text-left py-3 px-4 text-sm font-semibold rounded-lg transition-all ${activeTab === 'main' ? 'bg-white/10 text-white shadow-sm border border-white/20' : 'text-gray-400 hover:text-white hover:bg-white/5 border border-transparent'}`}
          >
            🕹️ Predictor
          </button>
          <button 
            onClick={() => setActiveTab('metrics')}
            className={`w-full text-left py-3 px-4 text-sm font-semibold rounded-lg transition-all ${activeTab === 'metrics' ? 'bg-white/10 text-white shadow-sm border border-white/20' : 'text-gray-400 hover:text-white hover:bg-white/5 border border-transparent'}`}
          >
            📊 Training Logs
          </button>
        </div>

        <div className="mt-auto text-xs text-gray-500 leading-relaxed">
          <p>React/ThreeJS Minimal Architecture</p>
        </div>
      </div>

      {/* MAIN LAYOUT AREA */}
      {activeTab === 'main' && (
        <div className="flex-1 flex flex-col p-8 gap-6 bg-[#05050a] animate-fade-in overflow-y-auto">
          
          {/* HEADER CONTROLS */}
          <div className="w-full bg-white/5 border border-white/10 rounded-2xl p-6 flex flex-col xl:flex-row items-center gap-6 shadow-lg">
            
            <div className="flex-1 w-full xl:w-auto">
              <label className="text-xs text-gray-400 uppercase tracking-widest mb-2 block">Select Original Sample</label>
              <select 
                className="w-full bg-black/40 border border-white/20 text-white p-3 rounded-lg outline-none focus:border-purple-500 transition-colors cursor-pointer"
                value={selectedCell}
                onChange={e => setSelectedCell(e.target.value)}
              >
                {cells.map(c => (
                  <option key={c.filename} value={c.filename}>
                    {c.type} (Anomaly Score: {c.score})
                  </option>
                ))}
              </select>
            </div>

            <button 
              onClick={handlePredict}
              disabled={loading}
              className="w-full xl:w-auto mt-6 xl:mt-0 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold py-4 px-12 rounded-lg shadow-[0_0_20px_rgba(124,58,237,0.4)] transition-all hover:scale-105 active:scale-95 disabled:opacity-50 disabled:pointer-events-none whitespace-nowrap ml-auto uppercase tracking-wide text-sm"
            >
              {loading ? "Running Neural Inference..." : "✨ Generate 3D Model"}
            </button>

          </div>

          {/* PREVIEW IMAGES ROW */}
          {previewImages && (
            <div className="w-full bg-white/5 border border-white/10 rounded-2xl p-8 flex flex-col border-dashed items-center justify-center shadow-sm">
              <h3 className="text-gray-400/80 font-light tracking-widest uppercase text-xs mb-8">Input 2D Micrographs</h3>
              <div className="flex flex-col md:flex-row gap-16 md:gap-24 items-center justify-center w-full">
                <div className="flex flex-col items-center">
                  <span className="text-xs text-gray-400 font-bold uppercase tracking-widest mb-4">Top Photo</span>
                  <img src={previewImages.top} className="w-48 h-48 border border-white/20 rounded shadow-2xl object-cover bg-black" style={{ imageRendering: 'pixelated' }} />
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-xs text-gray-400 font-bold uppercase tracking-widest mb-4">Bottom Photo</span>
                  <img src={previewImages.bottom} className="w-48 h-48 border border-white/20 rounded shadow-2xl object-cover bg-black" style={{ imageRendering: 'pixelated' }} />
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-xs text-gray-400 font-bold uppercase tracking-widest mb-4">Side Photo</span>
                  <img src={previewImages.side} className="w-48 h-48 border border-white/20 rounded shadow-2xl object-cover bg-black" style={{ imageRendering: 'pixelated' }} />
                </div>
              </div>
            </div>
          )}

          {/* 3D VIEWPORTS (ONLY SHOWN WHEN GENERATED) */}
          {(loading || data) && (
            <div className="flex flex-col w-full flex-1">
              {data && (
                <div className="w-full mb-4 bg-white/5 border border-white/10 rounded-xl p-4 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-3">
                  {metricCards.map((m: any) => (
                    <div key={m.key} className="flex flex-col" title={m.tooltip}>
                      <span className="text-xs text-gray-400 uppercase tracking-widest mb-1">{m.label}</span>
                      <span className={`text-xl font-mono leading-none ${m.color}`}>{m.value}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className="text-purple-400/50 text-xs text-center font-mono tracking-widest uppercase mb-4 mt-2">
                Pan: Right Click &nbsp;|&nbsp; Rotate: Left Click &nbsp;|&nbsp; Zoom: Scroll
              </div>
              <div className="w-full flex-1 flex flex-col md:flex-row items-start justify-center gap-8 pb-8">
          
          {/* BLOCK 1: AI PREDICTION */}
          <div className="w-full md:w-1/2 h-[65vh] flex flex-col bg-white/5 border border-white/10 rounded-2xl overflow-hidden relative shadow-2xl">
            <div className="absolute top-0 w-full p-5 bg-gradient-to-b from-purple-900/50 to-transparent z-10 pointer-events-none">
              <h4 className="text-purple-400 font-bold tracking-widest uppercase text-lg drop-shadow-md">AI Prediction</h4>
              <p className="text-white/60 text-xs mt-1">Neural Tri-View Reconstruction</p>
            </div>
            
            <div className="flex-1 relative cursor-move">
              {loading && (
                 <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20 bg-[#050510]/50 backdrop-blur-sm">
                   <div className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin"></div>
                 </div>
              )}
              {!data && !loading && (
                 <div className="absolute inset-0 flex items-center justify-center text-white/20 pointer-events-none text-xl lowercase font-light tracking-widest">
                   awaiting selection...
                 </div>
              )}
              {/* Отельный Canvas для Предсказания */}
              <Canvas camera={{ position: [0, 80, 100], fov: 45 }}>
                <color attach="background" args={['#050510']} />
                <ambientLight intensity={0.2} />
                <pointLight position={[10, 10, 10]} intensity={1.5} color="#8b5cf6" />
                <pointLight position={[-10, -10, -10]} intensity={1} color="#3b82f6" />
                <Environment preset="city" />
                <Stars radius={100} depth={50} count={1000} factor={4} fade />
                
                {data?.pred && (
                  <CellMesh vertices={data.pred.vertices} indices={data.pred.indices} color="#9333ea" />
                )}
                
                <SyncedControls />
              </Canvas>
            </div>
          </div>

          {/* BLOCK 2: GROUND TRUTH */}
          <div className="w-full md:w-1/2 h-[65vh] flex flex-col bg-white/5 border border-white/10 rounded-2xl overflow-hidden relative shadow-2xl">
            <div className="absolute top-0 w-full p-5 bg-gradient-to-b from-blue-900/50 to-transparent z-10 pointer-events-none">
              <h4 className="text-blue-400 font-bold tracking-widest uppercase text-lg drop-shadow-md">Ground Truth</h4>
              <p className="text-white/60 text-xs mt-1">Original Physical Organism</p>
            </div>
            
            <div className="flex-1 relative cursor-move">
              {!data && !loading && (
                 <div className="absolute inset-0 flex items-center justify-center text-white/20 pointer-events-none text-xl lowercase font-light tracking-widest">
                   awaiting selection...
                 </div>
              )}
              {/* Отельный Canvas для GT */}
              <Canvas camera={{ position: [0, 80, 100], fov: 45 }}>
                <color attach="background" args={['#050510']} />
                <ambientLight intensity={0.2} />
                <pointLight position={[10, 10, 10]} intensity={1.5} color="#8b5cf6" />
                <pointLight position={[-10, -10, -10]} intensity={1} color="#3b82f6" />
                <Environment preset="city" />
                <Stars radius={100} depth={50} count={1000} factor={4} fade />
                
                {data?.gt && (
                  <CellMesh vertices={data.gt.vertices} indices={data.gt.indices} color="#3b82f6" />
                )}
                
                <SyncedControls />
              </Canvas>
            </div>
          </div>
        </div>
      </div>
      )}
    </div>
    )}

      {/* METRICS TAB */}
      {activeTab === 'metrics' && (
        <div className="flex-1 overflow-y-auto p-12 bg-[#05050a] animate-fade-in flex flex-col gap-8">
          <div className="mb-4">
            <h2 className="text-3xl font-light text-white mb-2">Model Performance History</h2>
            <p className="text-gray-400">Reconstruction Autoencoder training over 50 epochs (BCE + Dice Hybrid Loss)</p>
          </div>

          <div className="w-full h-[400px] bg-white/5 border border-white/10 rounded-2xl p-6 shadow-2xl">
            <h3 className="text-lg font-semibold text-blue-300 mb-6">Loss Convergence (Train vs Test)</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={metricsHistory} margin={{ left: -20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" vertical={false} />
                <XAxis dataKey="epoch" stroke="#ffffff50" fontSize={12} tickMargin={10} />
                <YAxis stroke="#ffffff50" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0a0a16', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px' }} />
                <Line type="monotone" name="Train Loss" dataKey="train_loss" stroke="#3b82f6" strokeWidth={3} dot={false} />
                <Line type="monotone" name="Test Loss" dataKey="test_loss" stroke="#ef4444" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="w-full h-[400px] bg-white/5 border border-white/10 rounded-2xl p-6 shadow-2xl">
            <h3 className="text-lg font-semibold text-purple-300 mb-6">Accuracy Metrics (Dice & IoU)</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={metricsHistory} margin={{ left: -20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" vertical={false} />
                <XAxis dataKey="epoch" stroke="#ffffff50" fontSize={12} tickMargin={10} />
                <YAxis stroke="#ffffff50" fontSize={12} domain={[0.5, 1.0]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0a0a16', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px' }} />
                <Line type="monotone" name="Test Dice" dataKey="test_dice" stroke="#a855f7" strokeWidth={3} dot={false} />
                <Line type="monotone" name="Test IoU" dataKey="test_iou" stroke="#10b981" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
