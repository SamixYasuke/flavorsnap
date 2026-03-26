# 📊 Analytics Dashboard for FlavorSnap

A comprehensive analytics dashboard for administrators to monitor usage patterns, model performance, and user engagement metrics for the FlavorSnap food classification application.

## 🌟 Features

### 📈 Real-time Usage Statistics
- **Daily/Monthly Request Tracking**: Monitor API usage over time
- **Active User Analytics**: Track unique user engagement
- **Accuracy Metrics**: Monitor classification accuracy trends
- **Interactive Charts**: Visualize data with line charts and bar graphs

### 🤖 Model Performance Metrics
- **Accuracy Comparison**: Compare different model performances
- **Inference Time Tracking**: Monitor response times
- **Confidence Scores**: Track prediction confidence levels
- **Model Health Monitoring**: Real-time model status updates

### 👥 User Engagement Analytics
- **Food Classification Distribution**: Pie chart showing most popular food categories
- **User Behavior Patterns**: Analyze usage patterns
- **Peak Usage Times**: Identify high-traffic periods
- **Geographic Distribution** (future enhancement)

### 📊 Export Functionality
- **JSON Export**: Download complete analytics data
- **Date Range Filtering**: Export data for specific time periods
- **Custom Reports**: Generate tailored analytics reports
- **Automated Reporting** (future enhancement)

### 📅 Custom Date Range Filtering
- **Flexible Date Selection**: Filter data by custom date ranges
- **Quick Presets**: 7 days, 30 days, 90 days options
- **Real-time Updates**: Instantly refresh data based on selected range
- **Persistent Filters**: Remember user preferences

## 🏗️ Architecture

### Frontend Components
```
frontend/
├── pages/
│   ├── analytics.tsx              # Main analytics dashboard
│   └── api/
│       └── analytics.ts          # API endpoint for analytics data
├── components/
│   ├── AnalyticsCard.tsx         # Stats card component
│   ├── DateRangeFilter.tsx       # Date range filter
│   ├── UsageChart.tsx           # Usage statistics chart
│   ├── ModelPerformanceChart.tsx  # Model performance chart
│   ├── UserEngagementChart.tsx   # User engagement pie chart
│   └── RealTimeActivity.tsx      # Real-time activity feed
```

### Backend API Endpoints
```
ml-model-api/
├── app.py                       # Main Flask application
├── analytics.py                 # Analytics data management
└── requirements.txt             # Python dependencies
```

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 15.3.3 with React 19
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React
- **Styling**: TailwindCSS
- **TypeScript**: Full type safety

### Backend
- **API**: Flask with RESTful endpoints
- **CORS**: Flask-CORS for cross-origin requests
- **Data**: Mock data generation (easily replaceable with real database)

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ and pip
- Git

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd ml-model-api
pip install -r requirements.txt
python app.py
```

### Access the Dashboard
- Frontend: http://localhost:3000/analytics
- API: http://localhost:5000/analytics

## 📡 API Endpoints

### Get All Analytics Data
```http
GET /analytics
```
Returns comprehensive analytics data including usage stats, model performance, and user engagement.

### Get Usage Statistics
```http
GET /analytics/usage?start_date=2024-01-01&end_date=2024-01-31
```
Filter usage data by date range.

### Get Model Performance
```http
GET /analytics/performance
```
Returns model performance metrics.

### Get User Engagement
```http
GET /analytics/engagement
```
Returns user engagement analytics.

### Get Real-time Activity
```http
GET /analytics/activity
```
Returns recent activity feed.

### Export Analytics Data
```http
GET /analytics/export?start_date=2024-01-01&end_date=2024-01-31
```
Export analytics data for specified date range.

## 📊 Data Models

### Usage Statistics
```typescript
interface UsageStats {
  date: string;
  requests: number;
  users: number;
  accuracy: number;
}
```

### Model Performance
```typescript
interface ModelPerformance {
  model: string;
  accuracy: number;
  inferenceTime: number;
  confidence: number;
}
```

### User Engagement
```typescript
interface UserEngagement {
  category: string;
  value: number;
  color: string;
}
```

## 🎨 UI Components

### Stats Cards
- **Total Requests**: Shows total API requests with trend
- **Active Users**: Displays active user count with growth
- **Average Accuracy**: Model accuracy percentage
- **Response Time**: Average inference time

### Interactive Charts
- **Line Chart**: Usage trends over time
- **Bar Chart**: Model performance comparison
- **Pie Chart**: Food category distribution

### Real-time Activity Feed
- **Classification Requests**: Recent food classifications
- **Model Updates**: Model improvement notifications
- **System Alerts**: Performance and traffic alerts

## 🔧 Customization

### Adding New Metrics
1. Update `analytics.py` to include new data sources
2. Add corresponding API endpoints in `app.py`
3. Create new chart components in `frontend/components/`
4. Update the main dashboard in `analytics.tsx`

### Styling Changes
- Modify TailwindCSS classes in components
- Update color schemes in `analytics.py`
- Customize chart configurations in Recharts components

### Data Integration
Replace mock data generation in `analytics.py` with real database connections:
```python
# Example: Database integration
def get_usage_stats_from_db(self, start_date=None, end_date=None):
    query = "SELECT * FROM usage_stats WHERE date BETWEEN %s AND %s"
    # Execute query and return results
```

## 🧪 Testing

### Frontend Tests
```bash
cd frontend
npm run test
npm run test:coverage
```

### Backend Tests
```bash
cd ml-model-api
python -m pytest
python -m pytest --cov=app
```

## 📈 Future Enhancements

### Planned Features
- [ ] Real database integration (PostgreSQL/MongoDB)
- [ ] WebSocket support for real-time updates
- [ ] Advanced filtering and search
- [ ] User role-based access control
- [ ] Automated email reports
- [ ] Mobile-responsive optimizations
- [ ] Data export to CSV/PDF
- [ ] Advanced anomaly detection
- [ ] Geographic user analytics
- [ ] A/B testing framework

### Performance Optimizations
- [ ] Data caching with Redis
- [ ] Database query optimization
- [ ] Chart lazy loading
- [ ] API response compression

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check if frontend is running
curl http://localhost:3000

# Check if backend is running
curl http://localhost:5000/health
```

#### API Connection Issues
```bash
# Verify CORS settings
curl -H "Origin: http://localhost:3000" http://localhost:5000/analytics
```

#### Chart Rendering Issues
- Check browser console for JavaScript errors
- Verify Recharts installation
- Check data format consistency

## 🤝 Contributing

### Adding New Features
1. Create feature branch from `analytics-dashboard`
2. Implement changes with proper testing
3. Update documentation
4. Submit pull request with clear description

### Code Style
- Follow existing TypeScript patterns
- Use TailwindCSS for styling
- Maintain consistent naming conventions
- Add proper error handling

## 📄 License

This analytics dashboard is part of the FlavorSnap project and follows the same MIT License.

## 📞 Support

For issues related to the analytics dashboard:
- Create GitHub issue with detailed description
- Include screenshots for UI issues
- Provide browser and environment details

---

**Built with 💚 for comprehensive food classification analytics**
