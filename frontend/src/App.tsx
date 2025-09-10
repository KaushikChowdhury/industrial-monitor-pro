
import { useState } from 'react';
import CameraTile from './components/CameraTile';

export default function App() {
  const [cameras, setCameras] = useState<string[]>(['cam-1']); // Start with one camera by default

  const addCamera = () => {
    const newCameraId = `cam-${Date.now()}`;
    setCameras(prev => [...prev, newCameraId]);
  };

  const removeCamera = (idToRemove: string) => {
    setCameras(prev => prev.filter(id => id !== idToRemove));
  };

  return (
    <div className="min-h-screen text-zinc-100 bg-zinc-900">
      <div className="max-w-7xl mx-auto p-4 sm:p-6">
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Industrial Monitor</h1>
          <button 
            onClick={addCamera}
            className="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-4 py-2 rounded-lg shadow-md"
          >
            + Add Camera
          </button>
        </header>

        {cameras.length === 0 ? (
          <div className="text-center py-16 bg-zinc-800 rounded-lg">
            <p className="text-zinc-400">No cameras are active.</p>
            <p className="text-zinc-500 mt-2">Click 'Add Camera' to start monitoring.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {cameras.map(id => (
              <CameraTile key={id} id={id} onRemove={removeCamera} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
