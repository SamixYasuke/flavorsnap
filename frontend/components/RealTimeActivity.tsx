import React from 'react';
import { Eye, CheckCircle, AlertCircle } from 'lucide-react';

interface ActivityItem {
  id: string;
  type: 'classification' | 'model_update' | 'alert';
  title: string;
  description: string;
  timestamp: string;
}

interface RealTimeActivityProps {
  activities: ActivityItem[];
}

const RealTimeActivity: React.FC<RealTimeActivityProps> = ({ activities }) => {
  const getIcon = (type: string) => {
    switch (type) {
      case 'classification':
        return <Eye className="w-5 h-5 text-blue-500" />;
      case 'model_update':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'alert':
        return <AlertCircle className="w-5 h-5 text-orange-500" />;
      default:
        return <Eye className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Real-time Activity</h2>
      <div className="space-y-4">
        {activities.map((activity) => (
          <div key={activity.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-3">
              {getIcon(activity.type)}
              <div>
                <p className="font-medium">{activity.title}</p>
                <p className="text-sm text-gray-600">{activity.description}</p>
              </div>
            </div>
            <span className="text-sm text-gray-500">{activity.timestamp}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RealTimeActivity;
