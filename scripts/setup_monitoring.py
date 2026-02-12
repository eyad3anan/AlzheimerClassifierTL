"""
Setup Model Monitoring with Evidently
"""
import argparse
import os
import json
from datetime import datetime


def setup_monitoring(model_name, monitoring_endpoint=None):
    """Initialize monitoring configuration"""
    
    monitoring_config = {
        "model_name": model_name,
        "monitoring_enabled": True,
        "metrics_to_track": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "prediction_drift",
            "data_drift"
        ],
        "alert_thresholds": {
            "accuracy_drop": 0.05,
            "drift_score": 0.3
        },
        "monitoring_endpoint": monitoring_endpoint or "http://localhost:9090",
        "created_at": datetime.now().isoformat()
    }
    
    # Create monitoring directory
    os.makedirs('monitoring', exist_ok=True)
    
    # Save configuration
    config_path = 'monitoring/monitoring_config.json'
    with open(config_path, 'w') as f:
        json.dump(monitoring_config, f, indent=2)
    
    print(f"✅ Monitoring configuration created at {config_path}")
    print(f"Model: {model_name}")
    print(f"Endpoint: {monitoring_config['monitoring_endpoint']}")
    
    # Create Prometheus configuration
    prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'alzheimer-classifier'
    static_configs:
      - targets: ['streamlit-app:8501']
    
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
"""
    
    with open('monitoring/prometheus.yml', 'w') as f:
        f.write(prometheus_config)
    
    print("✅ Prometheus configuration created")
    
    return monitoring_config


def main():
    parser = argparse.ArgumentParser(description='Setup model monitoring')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model to monitor')
    parser.add_argument('--monitoring-endpoint', type=str,
                        help='Monitoring endpoint URL')
    
    args = parser.parse_args()
    
    setup_monitoring(args.model_name, args.monitoring_endpoint)


if __name__ == "__main__":
    main()
