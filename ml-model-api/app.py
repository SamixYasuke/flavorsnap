


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@track_inference
def predict():
    start_time = time.time()
    

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:


# Register all management endpoints
register_all_endpoints(app, model_registry, ab_test_manager, deployment_manager, model_validator)

@app.route('/analytics/usage', methods=['GET'])
def get_usage_stats():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data = analytics.get_usage_stats(start_date, end_date)
    return jsonify(data)

@app.route('/analytics/performance', methods=['GET'])
def get_model_performance():
    data = analytics.get_model_performance()
    return jsonify(data)

@app.route('/analytics/engagement', methods=['GET'])
def get_user_engagement():
    data = analytics.get_user_engagement()
    return jsonify(data)

@app.route('/analytics/activity', methods=['GET'])
def get_real_time_activity():
    data = analytics.get_real_time_activity()
    return jsonify(data)

@app.route('/analytics/stats', methods=['GET'])
def get_stats_cards():
    data = analytics.get_stats_cards()
    return jsonify(data)

@app.route('/analytics/export', methods=['GET'])
def export_analytics():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data = analytics.export_data(start_date, end_date)
    return jsonify(data)

@app.route('/analytics', methods=['GET'])
def get_all_analytics():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    data = {
        'usageStats': analytics.get_usage_stats(start_date, end_date),
        'modelPerformance': analytics.get_model_performance(),
        'userEngagement': analytics.get_user_engagement(),
        'statsCards': analytics.get_stats_cards(),
        'realTimeActivity': analytics.get_real_time_activity()
    }
    return jsonify(data)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'version': '1.0.0',
        'analytics_enabled': True
    })

if __name__ == '__main__':
