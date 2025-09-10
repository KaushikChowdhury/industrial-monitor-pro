import React, { useState } from 'react'
import Head from 'next/head'
import dynamic from 'next/dynamic'
import MetricsBar from '../components/MetricsBar'
import CameraTile from '../components/CameraTile'

// Lazy-load charts only on client to avoid Jest/SSR canvas issues
const HistoryChart = dynamic(() => import('../components/Charts').then(m => m.HistoryChart), { ssr: false, loading: () => null })

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

export default function Home() {
  const [tiles, setTiles] = useState<{id:string,title:string}[]>([
    { id:'cam1', title:'Camera 1' },
    { id:'cam2', title:'Camera 2' },
    { id:'cam3', title:'Camera 3' },
    { id:'cam4', title:'Camera 4' },
  ])
  const [threshold, setThreshold] = useState(85)

  const addTile = () => {
    const n = tiles.length + 1
    setTiles([...tiles, { id: `cam${n}`, title: `Camera ${n}` }])
  }

  const exportCsv = () => window.open(`${API}/api/export/csv`, '_blank')
  const downloadPdf = () => window.open(`${API}/api/report/pdf`, '_blank')

  const calibrate = async () => {
    const min_angle = Number(prompt('Min Angle (e.g. 0) ?', '0') || '0')
    const max_angle = Number(prompt('Max Angle (e.g. 270) ?', '270') || '270')
    const min_value = Number(prompt('Min Value (e.g. 0) ?', '0') || '0')
    const max_value = Number(prompt('Max Value (e.g. 100) ?', '100') || '100')
    await fetch(`${API}/api/calibrate`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ min_angle, max_angle, min_value, max_value })})
    alert('Calibration applied. Point the webcam at the dial and re-run.')
  }

  return (
    <>
      <Head>
        <title>Industrial Monitor Pro</title>
      </Head>
      <main className="max-w-7xl mx-auto p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">🎯 Industrial Monitor Pro</h1>
          <div className="flex gap-2">
            <button className="btn" onClick={calibrate}>Calibrate Dial</button>
            <button className="btn" onClick={exportCsv}>Export CSV</button>
            <button className="btn btn-primary" onClick={downloadPdf}>Download PDF</button>
            <button className="btn" onClick={addTile}>Add Camera</button>
          </div>
        </div>

        <MetricsBar threshold={threshold} setThreshold={setThreshold} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tiles.map((t) => (
            <CameraTile key={t.id} id={t.id} title={t.title} onRemove={() => setTiles(tiles.filter(x => x.id !== t.id))} />)
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tiles.slice(0,2).map((t)=> <HistoryChart key={t.id} cameraId={t.id} />)}
        </div>

        <div className="text-xs text-slate-400">
          ROI: Saves 2–3 hours/day of manual logging, early anomaly detection prevents failures, full audit trail for compliance.
        </div>
      </main>
    </>
  )
}