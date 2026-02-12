# 🧠 Alzheimer's Classifier - CI/CD & MLOps Setup

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git
- GitHub account
- Hugging Face account (for model deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd alzheimerclassifiertl-resnet50-transfer-learning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure GitHub Secrets**
   
   Go to your GitHub repository → Settings → Secrets and variables → Actions
   
   Add the following secrets:
   - `HF_TOKEN` - Your Hugging Face API token
   - `DOCKER_USERNAME` - Your Docker Hub username
   - `DOCKER_PASSWORD` - Your Docker Hub password/token

## 📋 CI/CD Pipeline Overview

The GitHub Actions workflow (`.github/workflows/ci-cd-mlops.yml`) includes:

### Pipeline Jobs

1. **Code Quality** - Linting and formatting checks
2. **Unit Tests** - Automated testing with coverage
3. **Model Training** - Train model with MLflow tracking
4. **Model Evaluation** - Generate metrics and visualizations
5. **Model Validation** - Quality gates (85% accuracy, 80% F1-score)
6. **Docker Build** - Containerize application
7. **Deploy to Hugging Face** - Upload model to HF Hub
8. **Setup Monitoring** - Initialize monitoring stack
9. **Notifications** - Send status updates

### Triggering the Pipeline

**Automatic Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`

**Manual Trigger:**
1. Go to Actions tab in GitHub
2. Select "CI/CD MLOps Pipeline"
3. Click "Run workflow"
4. Optionally check "Trigger model training"

**Trigger Training via Commit:**
```bash
git commit -m "[train] Update model architecture"
git push
```

## 🐳 Docker Usage

### Build and Run Locally

```bash
# Build the image
docker build -t alzheimer-classifier .

# Run the container
docker run -p 8501:8501 -e HF_TOKEN=your_token alzheimer-classifier
```

### Run Full MLOps Stack

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- **Streamlit App**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 🔬 MLOps Workflows

### Training a Model

```bash
python scripts/train_model.py \
  --data-dir Data/OriginalDataset/train \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --mlflow-tracking-uri http://localhost:5000
```

**Parameters:**
- `--data-dir`: Path to training data
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate for optimizer
- `--mlflow-tracking-uri`: MLflow server URL

### Evaluating a Model

```bash
python scripts/evaluate_model.py \
  --model-path models/AlzheimerClassifierTL.h5 \
  --test-data-path Data/OriginalDataset/test \
  --batch-size 32
```

**Outputs:**
- `outputs/metrics.json` - Performance metrics
- `outputs/confusion_matrix.png` - Confusion matrix
- `outputs/per_class_metrics.png` - Per-class performance

### Validating Model Quality

```bash
python scripts/validate_metrics.py \
  --metrics-file outputs/metrics.json \
  --min-accuracy 0.85 \
  --min-f1-score 0.80
```

Returns exit code 0 if validation passes, 1 if fails.

### Generating Evaluation Report

```bash
python scripts/generate_report.py \
  --metrics-path outputs/metrics.json \
  --output-path outputs/evaluation_report.html
```

Open `outputs/evaluation_report.html` in a browser to view the report.

### Uploading to Hugging Face

```bash
python scripts/upload_to_hf.py \
  --model-path models/AlzheimerClassifierTL.h5 \
  --repo-id YourUsername/AlzheimerClassifierTL \
  --token your_hf_token
```

### Setting Up Monitoring

```bash
python scripts/setup_monitoring.py \
  --model-name AlzheimerClassifierTL \
  --monitoring-endpoint http://localhost:9090
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Code Quality Checks

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .
pylint app.py
```

## 📊 Monitoring & Observability

### MLflow Tracking

1. Start MLflow server:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

2. Access UI at http://localhost:5000

3. View experiments, compare runs, and manage models

### Prometheus Metrics

1. Access Prometheus at http://localhost:9090
2. Query metrics: `up`, `model_accuracy`, `prediction_latency`
3. Set up alerts for model degradation

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Login with admin/admin
3. Add Prometheus datasource
4. Create dashboards for:
   - Model performance metrics
   - Prediction latency
   - Error rates
   - Data drift

## 📁 Project Structure

```
alzheimerclassifiertl-resnet50-transfer-learning/
├── .github/
│   └── workflows/
│       └── ci-cd-mlops.yml       # CI/CD pipeline
├── .streamlit/
│   └── secrets.toml              # Streamlit secrets
├── scripts/
│   ├── train_model.py            # Model training
│   ├── evaluate_model.py         # Model evaluation
│   ├── validate_metrics.py       # Quality gates
│   ├── upload_to_hf.py           # HF deployment
│   ├── setup_monitoring.py       # Monitoring setup
│   └── generate_report.py        # Report generation
├── tests/
│   └── test_preprocessing.py     # Unit tests
├── monitoring/
│   ├── prometheus.yml            # Prometheus config
│   └── monitoring_config.json    # Monitoring settings
├── models/                       # Trained models
├── outputs/                      # Evaluation outputs
├── Data/                         # Training data
├── app.py                        # Streamlit app
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
└── README_CICD.md               # This file
```

## 🔐 Security Best Practices

1. **Never commit secrets** - Use GitHub Secrets or environment variables
2. **Use .gitignore** - Exclude sensitive files and large datasets
3. **Rotate tokens regularly** - Update HF_TOKEN and DOCKER_PASSWORD periodically
4. **Limit token permissions** - Use read/write tokens only where needed
5. **Enable branch protection** - Require PR reviews before merging to main

## 🐛 Troubleshooting

### Pipeline Fails on Code Quality
```bash
# Fix formatting
black .
isort .

# Check for issues
flake8 . --count --statistics
```

### Docker Build Fails
```bash
# Check Dockerfile syntax
docker build --no-cache -t alzheimer-classifier .

# View build logs
docker build -t alzheimer-classifier . 2>&1 | tee build.log
```

### Model Validation Fails
- Check if test data is available
- Verify model file exists
- Lower quality thresholds temporarily for testing
- Review metrics.json for actual performance

### MLflow Connection Issues
```bash
# Check if MLflow server is running
curl http://localhost:5000/health

# Restart MLflow
docker-compose restart mlflow
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run code quality checks: `black . && isort . && flake8 .`
5. Commit with descriptive message
6. Push and create a pull request
7. Wait for CI/CD pipeline to pass

## 📄 License

[Your License Here]

## 👥 Authors

[Your Name/Team]

---

**Need Help?** Open an issue on GitHub or contact the maintainers.
