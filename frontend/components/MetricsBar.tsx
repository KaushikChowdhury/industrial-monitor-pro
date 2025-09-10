import React, { useEffect, useState } from 'react'

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

export default function MetricsBar({ threshold, setThreshold }: { threshold: number; setThreshold: (v: number)=>void }) {
  const [metrics, setMetrics] = useState<any>({})

  useEffect(() => {
    // Guard for SSR/tests where WebSocket may be unavailable
    if (typeof window === 'undefined' || !("WebSocket" in window)) return
    let ws: WebSocket | null = null
    try {
      ws = new WebSocket(API.replace(/^http/, 'ws') + '/ws')
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data as string)
          if (data.metrics) setMetrics(data.metrics)
          if (typeof data.threshold === 'number') setThreshold(data.threshold)
        } catch {}
      }
      ws.onerror = () => { /* suppress test noise */ }
    } catch {}
    return () => { try { ws?.close() } catch {} }
  }, [setThreshold])

  return (
    <div className="card p-4 flex items-center justify-between">
      <div className="flex gap-6 text-sm">
        <div>Active: <b>{metrics.active_cameras ?? 0}</b></div>
        <div>Avg: <b>{metrics.avg_reading ?? 0}</b></div>
        <div>Max: <b>{metrics.max_reading ?? 0}</b></div>
        <div>Anoms(24h): <b>{metrics.anomaly_count ?? 0}</b> ({metrics.anomaly_rate ?? 0}%)</div>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-300">Global Fallback Threshold</label>
        <input type="range" min={0} max={100} value={threshold}
          onChange={async (e)=>{
            const v = Number(e.target.value); setThreshold(v)
            await fetch(`${API}/api/threshold`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({threshold: v}) })
          }} />
        <span className="w-12 text-right">{threshold}</span>
      </div>
    </div>
  )
}