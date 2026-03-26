import { NextApiRequest, NextApiResponse } from 'next';

interface AnalyticsData {
  usageStats: Array<{
    date: string;
    requests: number;
    users: number;
    accuracy: number;
  }>;
  modelPerformance: Array<{
    model: string;
    accuracy: number;
    inferenceTime: number;
    confidence: number;
  }>;
  userEngagement: Array<{
    category: string;
    value: number;
    color: string;
  }>;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Generate mock analytics data
    const usageStats = [];
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      usageStats.push({
        date: date.toISOString().split('T')[0],
        requests: Math.floor(Math.random() * 500) + 100,
        users: Math.floor(Math.random() * 200) + 50,
        accuracy: Math.random() * 10 + 85
      });
    }

    const modelPerformance = [
      { model: 'ResNet18', accuracy: 94.2, inferenceTime: 234, confidence: 87.5 },
      { model: 'ResNet34', accuracy: 95.1, inferenceTime: 312, confidence: 89.2 },
      { model: 'EfficientNet', accuracy: 93.8, inferenceTime: 189, confidence: 86.1 }
    ];

    const userEngagement = [
      { category: 'Akara', value: 23, color: '#FF6B6B' },
      { category: 'Bread', value: 19, color: '#4ECDC4' },
      { category: 'Egusi', value: 17, color: '#45B7D1' },
      { category: 'Moi Moi', value: 21, color: '#96CEB4' },
      { category: 'Rice and Stew', value: 12, color: '#FFEAA7' },
      { category: 'Yam', value: 8, color: '#DDA0DD' }
    ];

    const analyticsData: AnalyticsData = {
      usageStats,
      modelPerformance,
      userEngagement
    };

    res.status(200).json(analyticsData);
  } catch (error) {
    console.error('Analytics API error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}
