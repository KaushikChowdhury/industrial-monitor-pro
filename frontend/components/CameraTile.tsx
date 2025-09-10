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

export default function CameraTile({ id, title, onRemove }: { id: string; title: string; onRemove: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [streaming, setStreaming] = useState(false)
  const [reading, setReading] = useState<number>(0)
  const [detected, setDetected] = useState(false)
  const [status, setStatus] = useState<FrameResp['status']>('UNKNOWN')
  const [imageB64, setImageB64] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number>(0)
  const [thr, setThr] = useState<CamThresholds>({ low: 85, med: 90, high: 95 })

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

  useEffect(() => {
    let timer: any
    if (streaming) {
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
            const res = await axios.post<FrameResp>(`${API}/api/process_frame?camera_id=${id}`, form, {
              headers: { 'Content-Type': 'multipart/form-data' },
            })
            const d = res.data
            setReading(d.reading)
            setDetected(d.detected)
            setStatus(d.status)
            setConfidence(d.confidence)
            setImageB64(d.annotated_image_b64)
            setThr(d.thresholds)
          } catch (e) { console.error(e) }
        }, 'image/jpeg', 0.85)
      }, 700)
    }
    return () => clearInterval(timer)
  }, [streaming, id])

  const startWebcam = async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      if (videoRef.current) { videoRef.current.srcObject = media; setStreaming(true) }
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
    await fetch(`${API}/api/camera/${id}/thresholds`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ low, med, high }) })
    setThr({ low, med, high })
  }

  const badgeClass = status === 'UNKNOWN' ? 'badge-unknown' : (status === 'NORMAL' ? 'badge-ok' : (status === 'LOW' ? 'badge-warn' : 'badge-crit'))

  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold">{title}</div>
          <div className={`badge ${badgeClass} mt-1`}>{status}</div>
        </div>
        <div className="space-x-2">
          <button className="btn" onClick={openThresholds}>Thresholds</button>
          {!streaming ? (
            <button className="btn btn-primary" onClick={startWebcam}>Start Webcam</button>
          ) : (
            <button className="btn" onClick={stopWebcam}>Stop</button>
          )}
          <button className="btn" onClick={onRemove}>Remove</button>
        </div>
      </div>

      <div className="relative">
        <video ref={videoRef} autoPlay playsInline className="w-full rounded-xl" />
        <canvas ref={canvasRef} className="hidden" />
        {imageB64 && (
          <img src={`data:image/jpeg;base64,${imageB64}`} className="overlay-img" alt="annotated overlay" />
        )}
      </div>

      <div className="grid grid-cols-4 gap-3 text-center">
        <div className="card p-3">
          <div className="text-2xl font-bold">{detected ? reading.toFixed(1) : '--'}</div>
          <div className="text-xs text-slate-400">Reading</div>
        </div>
        <div className="card p-3">
          <div className="text-2xl font-bold">{(confidence * 100).toFixed(0)}%</div>
          <div className="text-xs text-slate-400">Confidence</div>
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