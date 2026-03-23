import React from 'react';
import { useRouter } from 'next/router';
import { Home, BarChart3, Camera, Settings } from 'lucide-react';

const AnalyticsNavigation: React.FC = () => {
  const router = useRouter();

  const navigationItems = [

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <h1 className="text-xl font-bold text-gray-900">🍲 FlavorSnap</h1>
          </div>

        </div>
      </div>
    </nav>
  );
};

export default AnalyticsNavigation;
