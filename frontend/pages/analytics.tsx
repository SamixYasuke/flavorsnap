import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Users, Activity, Download, Calendar, Filter, RefreshCw, Eye, Clock, CheckCircle, AlertCircle } from 'lucide-react';

interface UsageStats {
  date: string;
  requests: number;
  users: number;
  accuracy: number;
}

interface ModelPerformance {
  model: string;
  accuracy: number;
  inferenceTime: number;
  confidence: number;
}

interface UserEngagement {
  category: string;
  value: number;
  color: string;
}

const AnalyticsDashboard: React.FC = () => {
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [usageData, setUsageData] = useState<UsageStats[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [userEngagement, setUserEngagement] = useState<UserEngagement[]>([]);

  // Mock data generation
  useEffect(() => {
    generateMockData();
  }, []);

  const generateMockData = () => {
    // Generate usage statistics
    const usage: UsageStats[] = [];
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      usage.push({
        date: date.toLocaleDateString(),
        requests: Math.floor(Math.random() * 500) + 100,
        users: Math.floor(Math.random() * 200) + 50,
        accuracy: Math.random() * 10 + 85
      });
    }
    setUsageData(usage);

    // Generate model performance data
    setModelPerformance([
      { model: 'ResNet18', accuracy: 94.2, inferenceTime: 234, confidence: 87.5 },
      { model: 'ResNet34', accuracy: 95.1, inferenceTime: 312, confidence: 89.2 },
      { model: 'EfficientNet', accuracy: 93.8, inferenceTime: 189, confidence: 86.1 }
    ]);

    // Generate user engagement data
    setUserEngagement([
      { category: 'Akara', value: 23, color: '#FF6B6B' },
      { category: 'Bread', value: 19, color: '#4ECDC4' },
      { category: 'Egusi', value: 17, color: '#45B7D1' },
      { category: 'Moi Moi', value: 21, color: '#96CEB4' },
      { category: 'Rice and Stew', value: 12, color: '#FFEAA7' },
      { category: 'Yam', value: 8, color: '#DDA0DD' }
    ]);
  };

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => {
      generateMockData();
      setIsLoading(false);
    }, 1000);
  };

  const handleExport = () => {
    const data = {
      usageData,
      modelPerformance,
      userEngagement,
      exportDate: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics-report-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const statsCards = [
    { title: 'Total Requests', value: '12,847', change: '+12.5%', icon: Activity, color: 'bg-blue-500' },
    { title: 'Active Users', value: '3,421', change: '+8.2%', icon: Users, color: 'bg-green-500' },
    { title: 'Avg Accuracy', value: '94.2%', change: '+2.1%', icon: CheckCircle, color: 'bg-purple-500' },
    { title: 'Response Time', value: '234ms', change: '-15ms', icon: Clock, color: 'bg-orange-500' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}

          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
              <p className="text-gray-600 mt-1">Monitor usage patterns, model performance, and user engagement</p>
            </div>

              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
in
              >
                <Download className="w-4 h-4" />
                Export Report
              </button>


        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          {statsCards.map((stat, index) => (
            <div key={index} className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                  <p className={`text-sm mt-2 ${stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
                    {stat.change}
                  </p>
                </div>
                <div className={`${stat.color} p-3 rounded-lg`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Date Range Filter */}

                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"

                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
n

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Usage Statistics */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Usage Statistics</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={usageData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="requests" stroke="#3B82F6" name="Requests" />
                <Line type="monotone" dataKey="users" stroke="#10B981" name="Active Users" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Model Performance */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Performance</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelPerformance}>
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

          {/* User Engagement */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Food Classification Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={userEngagement}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ category, value }) => `${category}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {userEngagement.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Real-time Activity */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Real-time Activity</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Eye className="w-5 h-5 text-blue-500" />
                  <div>
                    <p className="font-medium">Classification Request</p>
                    <p className="text-sm text-gray-600">Jollof Rice - 95.2% confidence</p>
                  </div>
                </div>
                <span className="text-sm text-gray-500">2 min ago</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <div>
                    <p className="font-medium">Model Update</p>
                    <p className="text-sm text-gray-600">Accuracy improved to 94.5%</p>
                  </div>
                </div>
                <span className="text-sm text-gray-500">15 min ago</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-orange-500" />
                  <div>
                    <p className="font-medium">High Traffic Alert</p>
                    <p className="text-sm text-gray-600">150+ requests in last hour</p>
                  </div>
                </div>
                <span className="text-sm text-gray-500">1 hour ago</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;
