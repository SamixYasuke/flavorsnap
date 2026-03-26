import { useRouter } from 'next/router';
import { Camera, Upload, Zap, BarChart3 } from 'lucide-react';

export default function Classify() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-blue-50 py-12">
      <div className="max-w-4xl mx-auto px-4">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Choose Classification Mode
          </h1>
          <p className="text-lg text-gray-600">
            Select how you'd like to classify your food images
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Single Image Classification */}
          <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-6">
              <Camera className="w-8 h-8 text-blue-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Single Image
            </h2>
            <p className="text-gray-600 mb-6">
              Upload and classify one image at a time. Perfect for quick predictions and immediate results.
            </p>
            <ul className="space-y-2 mb-6 text-sm text-gray-600">
              <li className="flex items-center">
                <Zap className="w-4 h-4 text-yellow-500 mr-2" />
                Instant results
              </li>
              <li className="flex items-center">
                <Zap className="w-4 h-4 text-yellow-500 mr-2" />
                Detailed confidence scores
              </li>
              <li className="flex items-center">
                <Zap className="w-4 h-4 text-yellow-500 mr-2" />
                Top 3 predictions
              </li>
            </ul>
            <button
              onClick={() => router.push('/single')}
              className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition"
            >
              Classify Single Image
            </button>
          </div>

          {/* Batch Image Classification */}
          <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-6">
              <Upload className="w-8 h-8 text-green-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Batch Processing
            </h2>
            <p className="text-gray-600 mb-6">
              Upload and classify multiple images simultaneously. Ideal for large datasets and efficient processing.
            </p>
            <ul className="space-y-2 mb-6 text-sm text-gray-600">
              <li className="flex items-center">
                <BarChart3 className="w-4 h-4 text-green-500 mr-2" />
                Process up to 50 images
              </li>
              <li className="flex items-center">
                <BarChart3 className="w-4 h-4 text-green-500 mr-2" />
                Progress tracking
              </li>
              <li className="flex items-center">
                <BarChart3 className="w-4 h-4 text-green-500 mr-2" />
                Export results (JSON/CSV)
              </li>
            </ul>
            <button
              onClick={() => router.push('/batch')}
              className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition"
            >
              Start Batch Processing
            </button>
          </div>
        </div>

        <div className="mt-12 text-center">
          <button
            onClick={() => router.push('/')}
            className="text-gray-600 hover:text-gray-800"
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
