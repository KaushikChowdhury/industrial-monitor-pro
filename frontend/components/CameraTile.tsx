import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

type CamThresholds = { low: number; med: number; high: number }

type FrameResp = {
  detected: boolean
  status: 'UNKNOWN' | 'NORMAL' | 'LOW' | 'MEDIUM' | 'HIGH'
  reading: number
  confidence: number
  annotated_image_b64: string
  is_anomaly: boolean
  thresholds: CamThresholds
}

const DEMO_URLS = [
  'https://tse1.mm.bing.net/th/id/OIP.vFkkFDeIN6FrbZpx78rZ2gHaE8?w=474&h=474&c=7&p=0',
  'https://tse1.mm.bing.net/th/id/OIP.PBA5fyzcbYtB0x44PryliAHaHa?w=474&h=474&c=7&p=0',
  'https://tse2.mm.bing.net/th/id/OIP.U0OY5NUaEtRy04TlZvOVWAHaHa?w=474&h=474&c=7&p=0',
  'https://tse3.mm.bing.net/th/id/OIP.YCowa3Npw3ddstrClABxIQHaFb?w=474&h=474&c=7&p=0'
]

export default function CameraTile({ id, title, onRemove }: { id: string; title: string; onRemove: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [mode, setMode] = useState<'webcam'|'image'>('webcam')
  const [imageUrl, setImageUrl] = useState<string>('')
  const [streaming, setStreaming] = useState(false)
  const [reading, setReading] = useState<number>(0)
  const [prevReading, setPrevReading] = useState<number | null>(null)
  const [prevTs, setPrevTs] = useState<number | null>(null)
  const [velocity, setVelocity] = useState<number>(0)
  const [detected, setDetected] = useState(false)
  const [status, setStatus] = useState<FrameResp['status']>('UNKNOWN')
  const [imageB64, setImageB64] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number>(0)
  const [thr, setThr] = useState<CamThresholds>({ low: 85, med: 90, high: 95 })
  const [deltaLimit, setDeltaLimit] = useState<number>(12) // value/sec considered fast

  // Fetch camera thresholds on mount
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API}/api/camera/${id}/thresholds`)
        if (r.ok) {
          const j = await r.json(); setThr({ low: j.low, med: j.med, high: j.high })
        }
      } catch {}
    })()
  }, [id])

  // Webcam loop → send frames
  useEffect(() => {
    let timer: any
    if (streaming && mode === 'webcam') {
      timer = setInterval(async () => {
        const canvas = canvasRef.current
        const video = videoRef.current
        if (!canvas || !video) return
        if (video.readyState < 2) return
        const ctx = canvas.getContext('2d')!
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)
        canvas.toBlob(async (blob) => {
          if (!blob) return
          const form = new FormData()
          form.append('file', blob, 'frame.jpg')
          try {
            const res = await axios.post<FrameResp>(`${API}/api/process_frame?camera_id=${id}&delta_limit=${encodeURIComponent(String(deltaLimit))}`, form, {
              headers: { 'Content-Type': 'multipart/form-data' },
            })
            handleResp(res.data)
          } catch (e) { console.error(e) }
        }, 'image/jpeg', 0.85)
      }, 600)
    }
    return () => clearInterval(timer)
  }, [streaming, id, mode])

  const handleResp = (d: FrameResp) => {
    setDetected(d.detected)
    setStatus(d.status)
    setConfidence(d.confidence)
    setImageB64(d.annotated_image_b64)
    setThr(d.thresholds)

    const now = performance.now()
    if (prevReading !== null && prevTs !== null) {
      const dt = Math.max(1, now - prevTs) / 1000
      const dv = d.reading - prevReading
      setVelocity(dv / dt)
    }
    setPrevReading(d.reading)
    setPrevTs(now)
    setReading(d.reading)
  }

  const startWebcam = async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      if (videoRef.current) { videoRef.current.srcObject = media; setStreaming(true); setMode('webcam') }
    } catch (e) { alert('Unable to access camera: ' + (e as any).message) }
  }

  const stopWebcam = () => {
    const video = videoRef.current
    if (video && video.srcObject) { (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop()); video.srcObject = null }
    setStreaming(false)
  }

  const openThresholds = async () => {
    const low = Number(prompt(`LOW threshold for ${title}`, String(thr.low)) || thr.low)
    const med = Number(prompt(`MED threshold for ${title}`, String(thr.med)) || thr.med)
    const high = Number(prompt(`HIGH threshold for ${title}`, String(thr.high)) || thr.high)
    await fetch(`${API}/api/camera/${id}/thresholds`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ low, med, high })
    })
    setThr({ low, med, high })
  }

  const loadImageViaServer = async (url: string) => {
    setMode('image')
    setImageUrl(url)
    try {
      const res = await axios.post<FrameResp>(`${API}/api/process_image_url?camera_id=${id}&delta_limit=${encodeURIComponent(String(deltaLimit))}&url=${encodeURIComponent(url)}`)
      handleResp(res.data)
    } catch (e) { alert('Failed to process image URL.'); console.error(e) }
  }

  const badgeClass = status === 'UNKNOWN' ? 'badge-unknown'
    : (status === 'NORMAL' ? 'badge-ok' : (status === 'LOW' ? 'badge-warn' : 'badge-crit'))
  const velLabel = `${velocity>=0?'+':''}${velocity.toFixed(1)}/s`

  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold">{title}</div>
          <div className={`badge ${badgeClass} mt-1`}>{status}</div>
        </div>
        <div className="space-x-2">
          <button className="btn" onClick={openThresholds}>Thresholds</button>
          <input className="input w-24" type="number" step="1" value={deltaLimit}
                 onChange={(e)=>setDeltaLimit(Number(e.target.value)||0)} title="Delta limit (value/sec)" />
          <span className="text-xs text-slate-400">Δ-limit</span>
          {!streaming ? (
            <>
              <button className="btn btn-primary" onClick={startWebcam}>Start Webcam</button>
              <button className="btn" onClick={()=>setMode('image')}>Image Mode</button>
            </>
          ) : (
            <button className="btn" onClick={stopWebcam}>Stop</button>
          )}
          <button className="btn" onClick={onRemove}>Remove</button>
        </div>
      </div>

      <div className="relative">
        <video ref={videoRef} autoPlay playsInline className="w-full rounded-xl"
               style={{display: mode==='webcam'?'block':'none'}} />
        <canvas ref={canvasRef} className="hidden" />
        {imageB64 && (
          <img src={`data:image/jpeg;base64,${imageB64}`} className="overlay-img" alt="annotated overlay" />
        )}
      </div>

      {mode==='image' && (
        <div className="flex items-center gap-2">
          <input className="input flex-1" placeholder="Paste dial image URL"
                 value={imageUrl} onChange={(e)=>setImageUrl(e.target.value)} />
          <button className="btn" onClick={()=>imageUrl && loadImageViaServer(imageUrl)}>Load URL</button>
          {DEMO_URLS.map((u,i)=> (
            <button key={i} className="btn" onClick={()=>loadImageViaServer(u)}>Demo {i+1}</button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-5 gap-3 text-center">
        <div className="card p-3">
          <div className="text-2xl font-bold">{detected ? reading.toFixed(1) : '--'}</div>
          <div className="text-xs text-slate-400">Reading</div>
        </div>
        <div className="card p-3">
          <div className="text-2xl font-bold">{(confidence * 100).toFixed(0)}%</div>
          <div className="text-xs text-slate-400">Confidence</div>
        </div>
        <div className="card p-3">
          <div className="text-2xl font-bold">{velLabel}</div>
          <div className="text-xs text-slate-400">Velocity</div>
        </div>
        <div className="card p-3">
          <div className="text-sm">LOW<br/><span className="font-bold">{thr.low}</span></div>
        </div>
        <div className="card p-3">
          <div className="text-sm">MED/HIGH<br/><span className="font-bold">{thr.med} / {thr.high}</span></div>
        </div>
      </div>
    </div>
  )
}
