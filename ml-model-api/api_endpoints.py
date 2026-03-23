"""
Additional API endpoints for model management, A/B testing, and deployment
"""

from flask import Flask, request, jsonify
import json
from datetime import datetime

# Add these endpoints to the existing app.py file

def register_management_endpoints(app, model_registry, ab_test_manager, deployment_manager, model_validator):
    """Register model management endpoints"""
    
    @app.route('/api/models', methods=['GET'])
    def list_models():
        """List all registered models"""
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        models = model_registry.list_models(active_only)
        
        return jsonify({
            'models': [
                {
                    'version': m.version,
                    'created_at': m.created_at,
                    'created_by': m.created_by,
                    'description': m.description,
                    'accuracy': m.accuracy,
                    'loss': m.loss,
                    'epochs_trained': m.epochs_trained,
                    'is_active': m.is_active,
                    'is_stable': m.is_stable,
                    'tags': m.tags,
                    'model_path': m.model_path
                }
                for m in models
            ]
        })
    
    @app.route('/api/models/<version>', methods=['GET'])
    def get_model(version):
        """Get specific model details"""
        model = model_registry.get_model(version)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'version': model.version,
            'created_at': model.created_at,
            'created_by': model.created_by,
            'description': model.description,
            'accuracy': model.accuracy,
            'loss': model.loss,
            'epochs_trained': model.epochs_trained,
            'dataset_version': model.dataset_version,
            'is_active': model.is_active,
            'is_stable': model.is_stable,
            'tags': model.tags,
            'hyperparameters': model.hyperparameters,
            'model_path': model.model_path,
            'model_hash': model.model_hash
        })
    
    @app.route('/api/models/register', methods=['POST'])
    def register_model():
        """Register a new model version"""
        data = request.get_json()
        
        required_fields = ['version', 'model_path', 'created_by', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        success = model_registry.register_model(
            version=data['version'],
            model_path=data['model_path'],
            created_by=data['created_by'],
            description=data['description'],
            accuracy=data.get('accuracy'),
            loss=data.get('loss'),
            epochs_trained=data.get('epochs_trained'),
            dataset_version=data.get('dataset_version'),
            tags=data.get('tags', []),
            hyperparameters=data.get('hyperparameters', {})
        )
        
        if success:
            return jsonify({'message': f'Model {data[\"version\"]} registered successfully'}), 201
        else:
            return jsonify({'error': 'Failed to register model'}), 500
    
    @app.route('/api/models/<version>/activate', methods=['POST'])
    def activate_model(version):
        """Activate a model version"""
        success = model_registry.activate_model(version)
        if success:
            return jsonify({'message': f'Model {version} activated successfully'})
        else:
            return jsonify({'error': 'Failed to activate model'}), 500
    
    @app.route('/api/models/<version>/validate', methods=['POST'])
    def validate_model(version):
        """Validate a model version"""
        try:
            result = model_validator.validate_model(version)
            return jsonify({
                'model_version': result.model_version,
                'validation_timestamp': result.validation_timestamp,
                'passed': result.passed,
                'overall_score': result.overall_score,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'avg_inference_time': result.avg_inference_time,
                'avg_confidence': result.avg_confidence,
                'model_integrity_passed': result.model_integrity_passed,
                'performance_regression_detected': result.performance_regression_detected,
                'error_messages': result.error_messages,
                'detailed_metrics': result.detailed_metrics,
                'confusion_matrix_path': result.confusion_matrix_path
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/<version>/deploy', methods=['POST'])
    def deploy_model(version):
        """Deploy a model version"""
        force = request.json.get('force', False) if request.json else False
        
        success = deployment_manager.deploy_model(version, force)
        if success:
            return jsonify({'message': f'Model {version} deployed successfully'})
        else:
            return jsonify({'error': 'Failed to deploy model'}), 500
    
    @app.route('/api/deployment/rollback', methods=['POST'])
    def rollback_model():
        """Rollback to a previous model version"""
        data = request.get_json()
        target_version = data.get('target_version')
        reason = data.get('reason', 'Manual rollback')
        
        if not target_version:
            return jsonify({'error': 'target_version is required'}), 400
        
        success = deployment_manager.rollback_model(target_version, reason)
        if success:
            return jsonify({'message': f'Rolled back to model {target_version}'})
        else:
            return jsonify({'error': 'Failed to rollback model'}), 500
    
    @app.route('/api/deployment/health', methods=['GET'])
    def deployment_health():
        """Get deployment health status"""
        model_version = request.args.get('model_version')
        health = deployment_manager.health_check(model_version)
        return jsonify(health)
    
    @app.route('/api/deployment/history', methods=['GET'])
    def deployment_history():
        """Get deployment history"""
        limit = request.args.get('limit', 50, type=int)
        history = deployment_manager.get_deployment_history(limit)
        return jsonify({'history': history})
    
    @app.route('/api/deployment/rollback-versions', methods=['GET'])
    def available_rollback_versions():
        """Get available rollback versions"""
        versions = deployment_manager.get_available_rollback_versions()
        return jsonify({'versions': versions})

def register_ab_testing_endpoints(app, ab_test_manager):
    """Register A/B testing endpoints"""
    
    @app.route('/api/ab-tests', methods=['GET'])
    def list_ab_tests():
        """List all A/B tests"""
        status = request.args.get('status')
        tests = ab_test_manager.list_tests(status)
        return jsonify({'tests': tests})
    
    @app.route('/api/ab-tests', methods=['POST'])
    def create_ab_test():
        """Create a new A/B test"""
        data = request.get_json()
        
        required_fields = ['model_a_version', 'model_b_version']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        try:
            test_id = ab_test_manager.create_test(
                model_a_version=data['model_a_version'],
                model_b_version=data['model_b_version'],
                traffic_split=data.get('traffic_split', 0.5),
                description=data.get('description', ''),
                min_sample_size=data.get('min_sample_size', 100),
                confidence_threshold=data.get('confidence_threshold', 0.95)
            )
            
            return jsonify({
                'test_id': test_id,
                'message': 'A/B test created successfully'
            }), 201
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ab-tests/<test_id>', methods=['GET'])
    def get_ab_test(test_id):
        """Get A/B test details"""
        try:
            summary = ab_test_manager.get_test_summary(test_id)
            return jsonify(summary)
        except Exception as e:
            return jsonify({'error': str(e)}), 404
    
    @app.route('/api/ab-tests/<test_id>/end', methods=['POST'])
    def end_ab_test(test_id):
        """End an A/B test"""
        data = request.get_json() or {}
        winner = data.get('winner')
        
        success = ab_test_manager.end_test(test_id, winner)
        if success:
            return jsonify({'message': f'A/B test {test_id} ended successfully'})
        else:
            return jsonify({'error': 'Failed to end A/B test'}), 500
    
    @app.route('/api/ab-tests/<test_id>/metrics', methods=['GET'])
    def get_ab_test_metrics(test_id):
        """Get A/B test metrics"""
        try:
            metrics_a, metrics_b = ab_test_manager.get_test_metrics(test_id)
            return jsonify({
                'model_a_metrics': {
                    'model_version': metrics_a.model_version,
                    'total_predictions': metrics_a.total_predictions,
                    'correct_predictions': metrics_a.correct_predictions,
                    'accuracy': metrics_a.accuracy,
                    'avg_confidence': metrics_a.avg_confidence,
                    'avg_processing_time': metrics_a.avg_processing_time,
                    'predictions_by_class': metrics_a.predictions_by_class
                },
                'model_b_metrics': {
                    'model_version': metrics_b.model_version,
                    'total_predictions': metrics_b.total_predictions,
                    'correct_predictions': metrics_b.correct_predictions,
                    'accuracy': metrics_b.accuracy,
                    'avg_confidence': metrics_b.avg_confidence,
                    'avg_processing_time': metrics_b.avg_processing_time,
                    'predictions_by_class': metrics_b.predictions_by_class
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def register_utility_endpoints(app):
    """Register utility endpoints"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Basic health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'  # Updated version with model management
        })
    
    @app.route('/api/classes', methods=['GET'])
    def get_classes():
        """Get supported food classes"""
        return jsonify({
            'classes': ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam'],
            'count': 6
        })
    
    @app.route('/api/validation/history', methods=['GET'])
    def validation_history():
        """Get validation history"""
        model_version = request.args.get('model_version')
        limit = request.args.get('limit', 50, type=int)
        
        # This would need access to model_validator
        # For now, return empty response
        return jsonify({'history': []})

# Function to register all endpoints
def register_all_endpoints(app, model_registry, ab_test_manager, deployment_manager, model_validator):
    """Register all management endpoints"""
    register_management_endpoints(app, model_registry, ab_test_manager, deployment_manager, model_validator)
    register_ab_testing_endpoints(app, ab_test_manager)
    register_utility_endpoints(app)
