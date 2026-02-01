# BioScopeAI Classifier Service

A microservice for automated classification of microscopic images using deep learning models. Built with Python, Kafka, and TensorFlow/Keras for real-time image analysis in medical and biological research.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Kafka](https://img.shields.io/badge/Kafka-async-green.svg)](https://kafka.apache.org/)
[![Tests](https://img.shields.io/badge/tests-57%20passing-success.svg)](tests/)

## 🔬 Features

- **Asynchronous Processing**: Event-driven architecture using Kafka for scalable image classification
- **Deep Learning**: TensorFlow/Keras models optimized for microscopic image analysis
- **Multi-class Classification**: Support for various multi-class images
- **Image Preprocessing**: Automated BGR→RGB conversion, resizing, and normalization
- **Model Caching**: LRU cache for efficient model loading and metadata management
- **Checksum Verification**: SHA256-based model integrity validation
- **Production Ready**: Docker containerization, comprehensive logging, and error handling
- **Well Tested**: 57 unit tests with pytest, 100% coverage on critical paths

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────────┐         ┌─────────────────┐
│  Kafka Topic    │         │  Classifier Service  │         │  Kafka Topic    │
│  (jobs)         │────────▶│                      │────────▶│  (results)      │
└─────────────────┘         │  ┌────────────────┐  │         └─────────────────┘
                            │  │  Consumer      │  │
                            │  └────────┬───────┘  │
                            │           │          │
                            │  ┌────────▼───────┐  │
                            │  │  Preprocessing │  │
                            │  └────────┬───────┘  │
                            │           │          │
                            │  ┌────────▼───────┐  │
                            │  │  ML Inference  │  │
                            │  └────────┬───────┘  │
                            │           │          │
                            │  ┌────────▼───────┐  │
                            │  │  Producer      │  │
                            │  └────────────────┘  │
                            └──────────────────────┘
```


## 📁 Project Structure

```
bioscopeai-classifier-service/
├── Dockerfile                              # Docker image definition
├── pyproject.toml                          # Project configuration and dependencies
├── README.md                               # Project documentation
│
├── bioscopeai_classifier_service/          # Main application package
│   ├── __init__.py
│   ├── main.py                            # Application entry point
│   │
│   ├── config/                            # Application configuration
│   │   ├── __init__.py
│   │   ├── config.py                      # Settings and environment variables
│   │   └── logging_config.py              # Logging configuration
│   │
│   ├── kafka/                             # Kafka integration
│   │   ├── __init__.py
│   │   ├── consumers/                     # Kafka consumers
│   │   │   ├── __init__.py
│   │   │   ├── base_consumer.py           # Base consumer class
│   │   │   └── classification_consumer.py # Classification job consumer
│   │   └── producers/                     # Kafka producers
│   │       ├── __init__.py
│   │       ├── base_producer.py           # Base producer class
│   │       └── result_producer.py         # Classification result producer
│   │
│   ├── model_processing/                  # ML model processing
│   │   ├── __init__.py
│   │   ├── inference.py                   # Model inference execution
│   │   ├── loader.py                      # Model and metadata loading
│   │   ├── model_processing.py            # Main processing logic
│   │   └── preprocess.py                  # Image preprocessing
│   │
│   └── services/                          # Business logic services
│       ├── __init__.py
│       └── classification_logic.py        # Image classification service
│
├── docs/                                  # Documentation and configurations
│   ├── bioscopeai-classifier-service-config.yaml         # Active configuration
│   ├── bioscopeai-classifier-service-config-example.yaml # Example configuration
│   └── supervisord/                       # Supervisord configuration
│       └── bioscopeai-classifier-service.conf
│
├── ml_models/                             # Machine learning models
│   ├── bioscopeai_classifier_model.keras  # Keras model file
│   ├── checksum.sha256                    # Model checksum for verification
│   └── metadata.json                      # Model metadata
│
├── tests/                                 # Test suite
│   ├── conftest.py                        # Pytest fixtures and configuration
│   ├── fixtures/                          # Test fixtures
│   │   └── test-config.yaml               # Test configuration
│   └── unit/                              # Unit tests
│       └── model_processing/              # Model processing tests
│           ├── test_inference.py          # Inference tests (15)
│           ├── test_loader.py             # Loader tests (13)
│           ├── test_model_processing.py   # Service tests (5)
│           └── test_preprocess.py         # Preprocessing tests (24)
│
└── scripts/                               # Utility scripts
    └── run_tests.sh                       # Test execution script
```

## 🔌 API / Message Format

### Input Message (Kafka Topic: classification-jobs)

```json
{
  "image_id": "uuid-string",
  "image_data": "base64-encoded-image",
  "model_name": "bioscopeai_classifier_model"
}
```

### Output Message (Kafka Topic: classification-results)

```json
{
  "image_id": "uuid-string",
  "label": "bone_cells_group",
  "confidence": 0.87,
  "model_name": "bioscopeai_classifier_model",
  "status": "success",
  "all_predictions": [
    {"label": "bone_cells_group", "confidence": 0.87},
    {"label": "rbc_group", "confidence": 0.08},
    {"label": "other", "confidence": 0.05}
  ]
}
```

## 🛠️ Tech Stack

- **Python 3.13**
- **TensorFlow 2.20 / Keras 3.12**: Deep learning framework
- **aiokafka**: Async Kafka client for Python
- **OpenCV 4.12**: Image processing
- **NumPy 2.0**: Numerical computing
- **Pydantic 2.x**: Data validation and settings management
- **Loguru**: Structured logging
- **pytest 8.4**: Testing framework

## 👤 Issues & contact

**Aleksander Białka**
- Email: aleksander.bialka@icloud.com
- GitHub: [@aleksanderbialka](https://github.com/aleksanderbialka)

**Note**: This service is part of the BioScopeAI ecosystem for automated microscopic image analysis.
