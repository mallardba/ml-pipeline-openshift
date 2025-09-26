# Requirements Documentation

This file explains each dependency in `requirements.txt` and why it's needed for the MLOps pipeline project.

## Core ML Libraries

### torch>=2.0.0
**What it is**: PyTorch - the primary deep learning framework
**Why we need it**: 
- Train neural networks for both predictive and generative AI models
- Provides GPU acceleration for faster training
- Industry standard for ML research and production
**Used for**: Model training, fine-tuning, and inference

### torchvision>=0.15.0
**What it is**: Computer vision utilities for PyTorch
**Why we need it**:
- Pre-trained models (ResNet, EfficientNet, etc.)
- Image preprocessing and augmentation
- Common CV datasets and transforms
**Used for**: Image-based ML tasks, transfer learning

### transformers>=4.30.0
**What it is**: Hugging Face Transformers library
**Why we need it**:
- Pre-trained language models (BERT, GPT, T5, etc.)
- Easy fine-tuning and inference APIs
- Tokenization and model utilities
**Used for**: Generative AI tasks, NLP models, text processing

### datasets>=2.12.0
**What it is**: Hugging Face Datasets library
**Why we need it**:
- Access to thousands of pre-processed datasets
- Efficient data loading and caching
- Built-in data preprocessing pipelines
**Used for**: Training data management, dataset versioning

### accelerate>=0.20.0
**What it is**: Hugging Face Accelerate for distributed training
**Why we need it**:
- Easy multi-GPU and distributed training
- Mixed precision training support
- Memory optimization
**Used for**: Scaling training across multiple GPUs/nodes

## Data Processing

### pandas>=2.0.0
**What it is**: Data manipulation and analysis library
**Why we need it**:
- Data cleaning and preprocessing
- Feature engineering
- Data exploration and analysis
**Used for**: Data preprocessing pipeline, feature extraction

### numpy>=1.24.0
**What it is**: Numerical computing library
**Why we need it**:
- Array operations and mathematical functions
- Foundation for most ML libraries
- Memory-efficient numerical computations
**Used for**: Data manipulation, mathematical operations

### scikit-learn>=1.3.0
**What it is**: Machine learning library with traditional algorithms
**Why we need it**:
- Preprocessing utilities (scalers, encoders)
- Model evaluation metrics
- Traditional ML algorithms for comparison
**Used for**: Data preprocessing, model evaluation, baseline models

### scipy>=1.10.0
**What it is**: Scientific computing library
**Why we need it**:
- Statistical functions
- Signal processing
- Optimization algorithms
**Used for**: Statistical analysis, optimization tasks

## Model Export & Serving

### onnx>=1.14.0
**What it is**: Open Neural Network Exchange format
**Why we need it**:
- Export PyTorch models to portable ONNX format
- Cross-platform model deployment
- Optimized inference runtime
**Used for**: Model export, cross-platform deployment

### onnxruntime>=1.15.0
**What it is**: ONNX model inference runtime
**Why we need it**:
- Fast inference for ONNX models
- CPU and GPU acceleration
- Production-ready serving
**Used for**: Model inference, production serving

### fastapi>=0.100.0
**What it is**: Modern web framework for building APIs
**Why we need it**:
- Fast, async API development
- Automatic OpenAPI documentation
- Built-in validation and serialization
**Used for**: Model serving API, inference endpoints

### uvicorn[standard]>=0.22.0
**What it is**: ASGI server for FastAPI
**Why we need it**:
- Production-ready ASGI server
- WebSocket support
- High-performance async serving
**Used for**: Running FastAPI applications

### pydantic>=2.0.0
**What it is**: Data validation and settings management
**Why we need it**:
- Request/response validation
- Configuration management
- Type safety
**Used for**: API validation, configuration

## MLOps & Monitoring

### mlflow>=2.5.0
**What it is**: ML lifecycle management platform
**Why we need it**:
- Experiment tracking and logging
- Model versioning and registry
- Model deployment and serving
**Used for**: Experiment tracking, model registry, deployment

### evidently>=0.2.0
**What it is**: ML model monitoring and drift detection
**Why we need it**:
- Data drift detection
- Model performance monitoring
- Automated alerts and reports
**Used for**: Model monitoring, drift detection

### prometheus-client>=0.17.0
**What it is**: Prometheus metrics client for Python
**Why we need it**:
- Custom metrics collection
- Integration with Prometheus monitoring
- Performance and business metrics
**Used for**: Custom metrics, monitoring integration

## Cloud Storage & APIs

### boto3>=1.28.0
**What it is**: AWS SDK for Python
**Why we need it**:
- S3 integration for model storage
- AWS service integration
- Cloud storage management
**Used for**: S3 model storage, AWS integration

### minio>=7.1.0
**What it is**: MinIO Python client for S3-compatible storage
**Why we need it**:
- Local S3-compatible storage
- Development and testing
- On-premises storage solutions
**Used for**: Local S3 storage, development environment

### requests>=2.31.0
**What it is**: HTTP library for Python
**Why we need it**:
- API calls and HTTP requests
- Integration with external services
- Simple HTTP client
**Used for**: API integrations, external service calls

## Utilities

### python-dotenv>=1.0.0
**What it is**: Load environment variables from .env files
**Why we need it**:
- Configuration management
- Environment-specific settings
- Security (keeping secrets out of code)
**Used for**: Configuration, environment management

### pyyaml>=6.0
**What it is**: YAML parser and emitter
**Why we need it**:
- Kubernetes manifest parsing
- Configuration file handling
- YAML-based settings
**Used for**: K8s manifests, configuration files

### click>=8.1.0
**What it is**: Command-line interface creation toolkit
**Why we need it**:
- CLI tools and scripts
- Command-line utilities
- User-friendly interfaces
**Used for**: CLI tools, automation scripts

### tqdm>=4.65.0
**What it is**: Progress bar library
**Why we need it**:
- Training progress visualization
- Long-running operation feedback
- User experience improvement
**Used for**: Progress bars, training visualization

## Optional GPU Support

### torch-audio>=2.0.0 (commented)
**What it is**: Audio processing for PyTorch
**Why you might need it**:
- Audio-based ML tasks
- Speech processing
- Audio data augmentation
**Used for**: Audio ML tasks (uncomment if needed)

### torchaudio>=2.0.0 (commented)
**What it is**: Audio utilities for PyTorch
**Why you might need it**:
- Audio data loading
- Audio preprocessing
- Audio model training
**Used for**: Audio processing (uncomment if needed)
