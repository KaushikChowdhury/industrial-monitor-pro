
import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Helper to determine status color
const getStatusColor = (status: string) => {
  switch (status) {
    case 'HIGH': return 'bg-red-500';
    case 'MEDIUM': return 'bg-orange-500';
    case 'LOW': return 'bg-yellow-500';
    case 'NORMAL': return 'bg-green-500';
    default: return 'bg-gray-500';
  }
};

export default function CameraTile({ id, onRemove }: { id: string; onRemove: (id: string) => void; }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streaming, setStreaming] = useState(false);
  const [reading, setReading] = useState<number | null>(null);
  const [status, setStatus] = useState('UNKNOWN');
  const [confidence, setConfidence] = useState(0);
  // The imageB64 state is no longer used for display, but we can keep it for potential future debugging
  const [imageB64, setImageB64] = useState('');
  const [thresholds, setThresholds] = useState({ low: 10, med: 20, high: 30 });

  // Fetch initial thresholds for this camera
  useEffect(() => {
    axios.get(`${API_BASE}/api/camera/${id}/thresholds`).then(res => {
      setThresholds(res.data);
    }).catch(console.error);
  }, [id]);

  // Main processing loop
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (streaming) {
      timer = setInterval(() => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video || video.readyState < 2) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
          if (!blob) return;
          const formData = new FormData();
          formData.append('file', blob, 'frame.jpg');
          try {
            const res = await axios.post(`${API_BASE}/api/process_frame?camera_id=${id}`, formData);
            const d = res.data;
            setReading(d.reading);
            setStatus(d.status);
            setConfidence(d.confidence);
            setImageB64(d.annotated_image_b64); // We still receive it, just don't display it
          } catch (e) {
            console.error('Frame processing failed:', e);
          }
        }, 'image/jpeg', 0.85);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [streaming, id]);

  const toggleStreaming = async () => {
    if (!streaming) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStreaming(true);
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
      setStreaming(false);
    }
  };

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setThresholds(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const saveThresholds = () => {
    axios.post(`${API_BASE}/api/camera/${id}/thresholds`, thresholds)
      .then(() => alert('Thresholds saved!'))
      .catch(err => alert(`Error saving thresholds: ${err.message}`));
  };

  return (
    <div className="bg-zinc-800 rounded-lg shadow-lg p-4 flex flex-col gap-4">
      {/* FIX: Always display the raw video feed and remove the annotated overlay logic */}
      <div className="relative h-48 bg-black rounded-md overflow-hidden">
        <video ref={videoRef} autoPlay playsInline className="w-full h-full object-contain" />
        <canvas ref={canvasRef} className="hidden" />
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-zinc-700 p-2 rounded">
          <div className="text-xs text-zinc-400">Reading</div>
          <div className="text-lg font-bold">{reading !== null ? reading.toFixed(1) : '---'}</div>
        </div>
        <div className={`${getStatusColor(status)} p-2 rounded`}>
          <div className="text-xs opacity-80">Status</div>
          <div className="text-lg font-bold">{status}</div>
        </div>
        <div className="bg-zinc-700 p-2 rounded">
          <div className="text-xs text-zinc-400">Confidence</div>
          <div className="text-lg font-bold">{(confidence * 100).toFixed(0)}%</div>
        </div>
      </div>

      <div className="flex gap-2 items-end">
        <div className="flex-1">
          <label className="text-xs text-zinc-400">Thresholds (L/M/H)</label>
          <div className="flex gap-1">
            <input type="number" name="low" value={thresholds.low} onChange={handleThresholdChange} className="w-full bg-zinc-900 rounded p-1 text-sm" />
            <input type="number" name="med" value={thresholds.med} onChange={handleThresholdChange} className="w-full bg-zinc-900 rounded p-1 text-sm" />
            <input type="number" name="high" value={thresholds.high} onChange={handleThresholdChange} className="w-full bg-zinc-900 rounded p-1 text-sm" />
          </div>
        </div>
        <button onClick={saveThresholds} className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-sm h-fit">Save</button>
      </div>

      <div className="flex justify-between items-center mt-2">
        <button onClick={toggleStreaming} className={`w-full mr-2 px-4 py-2 rounded ${streaming ? 'bg-red-600' : 'bg-green-600'}`}>
          {streaming ? 'Stop Camera' : 'Start Camera'}
        </button>
        <button onClick={() => onRemove(id)} className="bg-zinc-700 hover:bg-zinc-600 px-3 py-2 rounded" title="Remove Camera">🗑️</button>
      </div>
    </div>
  );
}
