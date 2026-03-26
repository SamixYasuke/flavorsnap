# Batch Image Processing Feature

## Overview

This feature adds comprehensive batch image processing capabilities to FlavorSnap, allowing users to upload and classify multiple food images simultaneously. The implementation includes a robust queue system, real-time progress tracking, and detailed result export functionality.

## Features

### ✅ Implemented Features

1. **Multi-file Upload Interface**
   - Drag-and-drop support for up to 50 images
   - File validation (type, size limits)
   - Visual preview of selected files
   - Individual file removal

2. **Batch Processing Queue**
   - Asynchronous processing with threading
   - Job management with unique IDs
   - Queue status monitoring
   - Concurrent processing support

3. **Progress Tracking**
   - Real-time progress updates
   - Individual file status tracking
   - Job status indicators (pending, processing, completed, failed, cancelled)
   - Processing statistics

4. **Results Summary and Export**
   - Comprehensive job summaries
   - Label distribution analytics
   - Performance metrics (confidence, processing time)
   - Export to JSON and CSV formats

5. **Error Handling**
   - Individual file error tracking
   - Graceful failure handling
   - Detailed error reporting
   - Job cancellation support

## Architecture

### Backend Components

#### 1. Batch Processor (`batch_processor.py`)
- **BatchProcessor Class**: Core batch processing logic
- **BatchJob Dataclass**: Job state management
- **BatchResult Dataclass**: Individual file results
- Threading-based queue processing
- Integration with existing model registry and A/B testing

#### 2. Batch Endpoints (`batch_endpoints.py`)
- **POST /api/batch/upload**: Create new batch job
- **GET /api/batch/status/{job_id}**: Get job status
- **GET /api/batch/results/{job_id}**: Get detailed results
- **GET /api/batch/summary/{job_id}**: Get job summary
- **GET /api/batch/export/{job_id}**: Export results
- **POST /api/batch/cancel/{job_id}**: Cancel job
- **GET /api/batch/jobs**: List all jobs
- **GET /api/batch/health**: Health check

#### 3. Integration Points
- Seamless integration with existing Flask app
- Uses existing model loading and prediction logic
- Compatible with A/B testing framework
- Maintains deployment health metrics

### Frontend Components

#### 1. Batch Processing Page (`pages/batch.tsx`)
- React component with TypeScript
- Drag-and-drop file upload using react-dropzone
- Real-time progress polling
- Results visualization and export
- Responsive design with Tailwind CSS

#### 2. Classification Hub (`pages/classify.tsx`)
- Navigation between single and batch processing
- Feature comparison
- User-friendly interface selection

#### 3. Single Image Page (`pages/single.tsx`)
- Traditional single image classification
- Fallback for users preferring individual processing

## API Documentation

### Create Batch Job
```http
POST /api/batch/upload
Content-Type: multipart/form-data

files: [File, File, ...] (max 50 files, 10MB each)
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "total_files": 5,
  "message": "Batch job created with 5 files"
}
```

### Get Job Status
```http
GET /api/batch/status/{job_id}
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:05Z",
  "total_files": 5,
  "processed_files": 2,
  "failed_files": 0,
  "progress_percentage": 40.0
}
```

### Get Job Results
```http
GET /api/batch/results/{job_id}
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "summary": { ... },
  "results": [
    {
      "filename": "image1.jpg",
      "label": "Jollof Rice",
      "confidence": 0.95,
      "all_predictions": [...],
      "processing_time": 0.234,
      "model_version": "v1.0.0"
    }
  ],
  "errors": []
}
```

### Export Results
```http
GET /api/batch/export/{job_id}?format=json|csv
```

Downloads the results in the specified format.

## Installation and Setup

### Backend Setup

1. **Install Dependencies**
   ```bash
   cd ml-model-api
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Verify Batch Endpoints**
   ```bash
   curl http://localhost:5000/api/batch/health
   ```

### Frontend Setup

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Environment Configuration**
   Create `.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:5000
   ```

3. **Run the Frontend**
   ```bash
   npm run dev
   ```

## Usage

### For Users

1. **Navigate to Batch Processing**
   - Go to the FlavorSnap homepage
   - Click "Get Started"
   - Select "Batch Processing"

2. **Upload Images**
   - Drag and drop images or click to select
   - Supports JPEG, PNG, WebP, GIF formats
   - Maximum 50 files per batch, 10MB per file

3. **Monitor Progress**
   - Real-time progress bar
   - Individual file status
   - Processing statistics

4. **View Results**
   - Detailed classification results
   - Confidence scores and predictions
   - Label distribution charts

5. **Export Data**
   - Download results as JSON or CSV
   - Include all metadata and predictions

### For Developers

1. **API Integration**
   ```python
   import requests
   
   # Create batch job
   files = [('files', open('image1.jpg', 'rb'))]
   response = requests.post('http://localhost:5000/api/batch/upload', files=files)
   job_id = response.json()['job_id']
   
   # Check status
   status = requests.get(f'http://localhost:5000/api/batch/status/{job_id}')
   ```

2. **Custom Processing**
   - Extend `BatchProcessor` class
   - Add custom validation logic
   - Implement additional export formats

## Performance Considerations

### Backend Optimization
- **Threading**: Uses background thread for non-blocking processing
- **Memory Management**: Efficient file handling and cleanup
- **Error Isolation**: Individual file failures don't affect batch
- **Scalability**: Queue-based architecture supports scaling

### Frontend Optimization
- **Polling**: Efficient status polling with 2-second intervals
- **Memory**: Lazy loading of results
- **UI**: Responsive design with loading states
- **Cancellation**: Proper cleanup of intervals and requests

### Resource Limits
- **Files per Batch**: 50 (configurable)
- **File Size**: 10MB per file (configurable)
- **Supported Formats**: JPEG, PNG, WebP, GIF
- **Concurrent Jobs**: Limited by system resources

## Error Handling

### File Validation Errors
- Invalid file types
- Size limit exceeded
- Corrupted files

### Processing Errors
- Model loading failures
- Image processing errors
- Network issues

### System Errors
- Memory constraints
- Disk space limitations
- Thread pool exhaustion

## Testing

### Unit Tests
```bash
cd ml-model-api
python -m pytest tests/test_batch_processor.py
```

### Integration Tests
```bash
cd ml-model-api
python -m pytest tests/test_batch_endpoints.py
```

### Frontend Tests
```bash
cd frontend
npm run test
```

## Monitoring and Logging

### Backend Metrics
- Job creation rate
- Processing time distribution
- Error rates by type
- Queue depth monitoring

### Health Checks
```bash
curl http://localhost:5000/api/batch/health
```

Response:
```json
{
  "status": "healthy",
  "current_job_id": "uuid-string",
  "queue_size": 0,
  "total_jobs": 15,
  "processing_thread_alive": true
}
```

## Future Enhancements

### Planned Features
1. **WebSocket Integration**: Real-time updates without polling
2. **Advanced Filtering**: Filter results by confidence, label, etc.
3. **Batch Comparison**: Compare results across different models
4. **Cloud Storage**: Integration with S3, Google Cloud Storage
5. **Batch Scheduling**: Schedule batch jobs for specific times

### Performance Improvements
1. **GPU Acceleration**: Leverage GPU for batch processing
2. **Parallel Processing**: Multi-threaded image processing
3. **Caching**: Result caching for duplicate images
4. **Compression**: Client-side image compression

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Increase system memory
   - Enable memory monitoring

2. **Slow Processing**
   - Check model loading status
   - Monitor CPU usage
   - Consider GPU acceleration

3. **Upload Failures**
   - Check file size limits
   - Verify network connectivity
   - Review server logs

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Code Style
- Python: PEP 8 compliant
- TypeScript: ESLint + Prettier
- Comments: Docstrings for all functions

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit PR with detailed description

## License

This feature is part of the FlavorSnap project and follows the same MIT License.

## Support

For issues and questions:
- GitHub Issues: [Create new issue](https://github.com/akordavid373/flavorsnap/issues)
- Documentation: [FlavorSnap Wiki](https://github.com/akordavid373/flavorsnap/wiki)

---

**Implementation Status**: ✅ Complete and Ready for Production

This batch processing feature significantly enhances the FlavorSnap platform by enabling efficient handling of large datasets while maintaining the high-quality classification accuracy and user experience that users expect.
