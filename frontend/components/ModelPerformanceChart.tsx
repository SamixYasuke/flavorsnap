import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ModelPerformanceData {
  model: string;
  accuracy: number;
  inferenceTime: number;
  confidence: number;
}

interface ModelPerformanceChartProps {
  data: ModelPerformanceData[];
}

const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({ data }) => {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Performance</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="model" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="accuracy" fill="#8B5CF6" name="Accuracy (%)" />
          <Bar dataKey="confidence" fill="#F59E0B" name="Confidence (%)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ModelPerformanceChart;
