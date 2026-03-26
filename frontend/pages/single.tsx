import { useState } from 'react';
import { useRouter } from 'next/router';

export default function SingleImage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setResult(null);
      setError('');
    }
  };

  const processImage = async () => {
    if (!selectedFile) return;

    setProcessing(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'Processing failed');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-2xl mx-auto px-4">
        <div className="mb-8">
          <button
            onClick={() => router.push('/classify')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Classification Options
          </button>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Single Image Classification</h1>
          <p className="text-gray-600">Upload a food image to get instant classification results</p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Food Image
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          {preview && (
            <div className="mb-6">
              <img
                src={preview}
                alt="Preview"
                className="w-full h-64 object-cover rounded-lg"
              />
            </div>
          )}

          {selectedFile && (
            <button
              onClick={processImage}
              disabled={processing}
              className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              {processing ? 'Processing...' : 'Classify Image'}
            </button>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="text-red-700">{error}</div>
            </div>
          )}

          {result && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="text-lg font-semibold text-green-800 mb-2">Classification Result</h3>
              <div className="space-y-2">
                <div>
                  <span className="font-medium">Predicted Dish:</span> {result.label}
                </div>
                <div>
                  <span className="font-medium">Confidence:</span> {result.confidence}%
                </div>
                <div>
                  <span className="font-medium">Processing Time:</span> {result.processing_time}s
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
