import { useState, useCallback } from 'react';
import { useRouter } from 'next/router';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Upload, X, CheckCircle, AlertCircle, Download, RefreshCw, Clock, FileText, BarChart3 } from 'lucide-react';

interface BatchJob {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  progress_percentage: number;
  errors_count: number;
}

interface BatchResult {
  filename: string;
  label: string;
  confidence: number;
  all_predictions: Array<{ label: string; confidence: number }>;
  processing_time: number;
  model_version: string;
  test_id?: string;
  error?: string;
}

interface BatchSummary {
  job_id: string;
  status: string;
  created_at: string;
  completed_at?: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  success_rate: number;
  label_distribution: Record<string, number>;
  average_confidence: number;
  total_processing_time: number;
  average_processing_time: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export default function BatchProcessing() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [currentJob, setCurrentJob] = useState<BatchJob | null>(null);
  const [jobResults, setJobResults] = useState<BatchResult[]>([]);
  const [jobSummary, setJobSummary] = useState<BatchSummary | null>(null);
  const [error, setError] = useState<string>('');
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
    setError('');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.gif']
    },
    maxFiles: 50,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const startBatchProcessing = async () => {
    if (files.length === 0) {
      setError('Please select at least one file');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios.post(`${API_BASE_URL}/api/batch/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const job = response.data;
      setCurrentJob(job);
      setFiles([]);
      
      // Start polling for job status
      startPolling(job.job_id);
      
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to start batch processing');
      setUploading(false);
    }
  };

  const startPolling = (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/batch/status/${jobId}`);
        const job = response.data;
        setCurrentJob(job);

        if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
          clearInterval(interval);
          setPollingInterval(null);
          setUploading(false);
          
          if (job.status === 'completed') {
            await fetchJobResults(jobId);
            await fetchJobSummary(jobId);
          }
        }
      } catch (err) {
        console.error('Error polling job status:', err);
        clearInterval(interval);
        setPollingInterval(null);
        setUploading(false);
      }
    }, 2000);

    setPollingInterval(interval);
  };

  const fetchJobResults = async (jobId: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/batch/results/${jobId}`);
      setJobResults(response.data.results);
    } catch (err) {
      console.error('Error fetching job results:', err);
    }
  };

  const fetchJobSummary = async (jobId: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/batch/summary/${jobId}`);
      setJobSummary(response.data);
    } catch (err) {
      console.error('Error fetching job summary:', err);
    }
  };

  const exportResults = async (format: 'json' | 'csv') => {
    if (!currentJob) return;

    try {
      const response = await axios.get(`${API_BASE_URL}/api/batch/export/${currentJob.job_id}`, {
        params: { format },
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `batch_results_${currentJob.job_id}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting results:', err);
      setError('Failed to export results');
    }
  };

  const cancelJob = async () => {
    if (!currentJob) return;

    try {
      await axios.post(`${API_BASE_URL}/api/batch/cancel/${currentJob.job_id}`);
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
      setUploading(false);
    } catch (err) {
      console.error('Error cancelling job:', err);
      setError('Failed to cancel job');
    }
  };

  const resetJob = () => {
    setCurrentJob(null);
    setJobResults([]);
    setJobSummary(null);
    setError('');
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      case 'processing':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'cancelled':
        return <X className="w-5 h-5 text-gray-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      case 'processing':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="mb-8">
          <button
            onClick={() => router.push('/')}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Home
          </button>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Batch Image Processing</h1>
          <p className="text-gray-600">Upload and classify multiple food images simultaneously</p>
        </div>

        {/* File Upload Section */}
        {!currentJob && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-blue-400 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              {isDragActive ? (
                <p className="text-blue-600">Drop the files here...</p>
              ) : (
                <div>
                  <p className="text-gray-600 mb-2">
                    Drag & drop food images here, or click to select files
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports: JPEG, PNG, WebP, GIF (Max 50 files, 10MB each)
                  </p>
                </div>
              )}
            </div>

            {files.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-3">Selected Files ({files.length})</h3>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-gray-50 p-3 rounded"
                    >
                      <span className="text-sm text-gray-700 truncate flex-1">
                        {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-red-500 hover:text-red-700 ml-2"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  onClick={startBatchProcessing}
                  disabled={uploading || files.length === 0}
                  className="mt-4 w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
                >
                  {uploading ? 'Starting...' : 'Start Batch Processing'}
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <span className="text-red-700">{error}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Job Status Section */}
        {currentJob && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Job Status</h2>
              <div className="flex items-center space-x-2">
                {getStatusIcon(currentJob.status)}
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(currentJob.status)}`}>
                  {currentJob.status.charAt(0).toUpperCase() + currentJob.status.slice(1)}
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Progress</span>
                  <span>{currentJob.processed_files + currentJob.failed_files} / {currentJob.total_files} files</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${currentJob.progress_percentage}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Processed:</span>
                  <span className="ml-2 font-medium">{currentJob.processed_files}</span>
                </div>
                <div>
                  <span className="text-gray-600">Failed:</span>
                  <span className="ml-2 font-medium text-red-600">{currentJob.failed_files}</span>
                </div>
                <div>
                  <span className="text-gray-600">Job ID:</span>
                  <span className="ml-2 font-mono text-xs">{currentJob.job_id.slice(0, 8)}...</span>
                </div>
                <div>
                  <span className="text-gray-600">Created:</span>
                  <span className="ml-2">{new Date(currentJob.created_at).toLocaleTimeString()}</span>
                </div>
              </div>

              {currentJob.status === 'processing' && (
                <button
                  onClick={cancelJob}
                  className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition"
                >
                  Cancel Job
                </button>
              )}

              {(currentJob.status === 'completed' || currentJob.status === 'failed' || currentJob.status === 'cancelled') && (
                <button
                  onClick={resetJob}
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
                >
                  Start New Batch
                </button>
              )}
            </div>
          </div>
        )}

        {/* Results Summary */}
        {jobSummary && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold flex items-center">
                <BarChart3 className="w-6 h-6 mr-2" />
                Results Summary
              </h2>
              <div className="flex space-x-2">
                <button
                  onClick={() => exportResults('json')}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition flex items-center"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export JSON
                </button>
                <button
                  onClick={() => exportResults('csv')}
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition flex items-center"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Export CSV
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{jobSummary.success_rate.toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Success Rate</div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{jobSummary.average_confidence.toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Avg Confidence</div>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{jobSummary.total_processing_time.toFixed(2)}s</div>
                <div className="text-sm text-gray-600">Total Time</div>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">{jobSummary.average_processing_time.toFixed(2)}s</div>
                <div className="text-sm text-gray-600">Avg Time/Image</div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Label Distribution</h3>
              <div className="space-y-2">
                {Object.entries(jobSummary.label_distribution).map(([label, count]) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-gray-700">{label}</span>
                    <div className="flex items-center">
                      <div className="w-32 bg-gray-200 rounded-full h-2 mr-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${(count / jobSummary.processed_files) * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium w-8">{count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Detailed Results */}
        {jobResults.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Detailed Results</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Filename</th>
                    <th className="text-left py-2">Prediction</th>
                    <th className="text-left py-2">Confidence</th>
                    <th className="text-left py-2">Processing Time</th>
                    <th className="text-left py-2">Model Version</th>
                  </tr>
                </thead>
                <tbody>
                  {jobResults.map((result, index) => (
                    <tr key={index} className="border-b">
                      <td className="py-2 truncate max-w-xs">{result.filename}</td>
                      <td className="py-2 font-medium">{result.label}</td>
                      <td className="py-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          result.confidence > 0.8 ? 'bg-green-100 text-green-800' :
                          result.confidence > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2">{result.processing_time.toFixed(3)}s</td>
                      <td className="py-2 font-mono text-xs">{result.model_version}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
