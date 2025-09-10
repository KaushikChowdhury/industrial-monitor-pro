import React, { useEffect, useState } from 'react'
import { Line } from 'react-chartjs-2'
import { Chart, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip } from 'chart.js'
Chart.register(LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip)

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

export function HistoryChart({ cameraId }: { cameraId: string }) {
  const [labels, setLabels] = useState<string[]>([])
  const [values, setValues] = useState<number[]>([])
  const [anoms, setAnoms] = useState<number[]>([])

  const load = async () => {
    const res = await fetch(`${API}/api/readings/${cameraId}?hours=4`)
    const json = await res.json()
    const L: string[] = []
    const V: number[] = []
    const A: number[] = []
    json.readings.forEach((r: any) => {
      L.push(new Date(r.timestamp).toLocaleTimeString())
      V.push(r.reading)
      A.push(r.is_anomaly ? r.reading : NaN)
    })
    setLabels(L); setValues(V); setAnoms(A)
  }

  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t) }, [cameraId])

  return (
    <div className="card p-4">
      <div className="font-semibold mb-2">{cameraId} – Last 4h</div>
      <Line data={{
        labels,
        datasets:[
          { label:'Reading', data: values, borderWidth: 2 },
          { label:'Anomaly', data: anoms, pointRadius: 5, borderWidth: 0 }
        ]
      }} options={{ responsive:true, plugins:{ legend:{ labels:{ color:'#cbd5e1' } }}, scales:{ x:{ ticks:{ color:'#94a3b8' } }, y:{ ticks:{ color:'#94a3b8' }, min:0, max:100 } } }} />
    </div>
  )
}