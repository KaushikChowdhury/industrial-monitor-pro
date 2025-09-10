import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

type CamThresholds = { low: number; med: number; high: number }

type FrameResp = {
  detected: boolean
  status: 'UNKNOWN' | 'NORMAL' | 'LOW' | 'MEDIUM' | 'HIGH'
  reading: number
  confidence: number
  is_anomaly: boolean
  thresholds: CamThresholds
  gauge_bbox: [number, number, number, number] | null
  pointer_bbox: [number, number, number, number] | null
}

export default function CameraTile({ id, title, onRemove }: { id: string; title: string; onRemove: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [streaming, setStreaming] = useState(false)
  const [reading, setReading] = useState<number>(0)
  const [prevReading, setPrevReading] = useState<number | null>(null)
  const [prevTs, setPrevTs] = useState<number | null>(null)
  const [velocity, setVelocity] = useState<number>(0)
  const [detected, setDetected] = useState(false)
  const [status, setStatus] = useState<FrameResp['status']>('UNKNOWN')
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
    if (streaming) {
      timer = setInterval(async () => {
        const canvas = canvasRef.current
        const video = videoRef.current
        if (!canvas || !video || video.readyState < 2) return

        // Ensure canvas dimensions match video
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        
        // Draw video to a temporary canvas to create a blob
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

        tempCanvas.toBlob(async (blob) => {
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
  }, [streaming, id, deltaLimit])

  const handleResp = (d: FrameResp) => {
    setDetected(d.detected)
    setStatus(d.status)
    setConfidence(d.confidence)
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

    // Drawing logic
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.readyState < 2) return;

    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (d.gauge_bbox) {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        const [x0, y0, x1, y1] = d.gauge_bbox;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.fillStyle = "#00FF00";
        ctx.font = "16px sans-serif";
        ctx.fillText(d.reading.toFixed(1), x0, y0 - 5);
    }
    if (d.pointer_bbox) {
        ctx.strokeStyle = "#FFEB3B";
        ctx.lineWidth = 2;
        const [x0, y0, x1, y1] = d.pointer_bbox;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
  }

  const startWebcam = async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      if (videoRef.current) { videoRef.current.srcObject = media; setStreaming(true); }
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
            <button className="btn btn-primary" onClick={startWebcam}>Start Webcam</button>
          ) : (
            <button className="btn" onClick={stopWebcam}>Stop</button>
          )}
          <button className="btn" onClick={onRemove}>Remove</button>
        </div>
      </div>

      <div className="relative bg-black rounded-xl">
        <video ref={videoRef} autoPlay playsInline className="hidden" />
        <canvas ref={canvasRef} className="w-full rounded-xl" />
      </div>

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
