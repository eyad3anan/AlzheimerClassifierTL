"""
Generate Evaluation Report
"""
import argparse
import json
import os
from datetime import datetime


def generate_html_report(metrics_path, output_path):
    """Generate HTML evaluation report"""
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    report = data.get('classification_report', {})
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Alzheimer's Classifier - Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">{metrics.get('test_accuracy', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test Precision</div>
            <div class="metric-value">{metrics.get('test_precision', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test Recall</div>
            <div class="metric-value">{metrics.get('test_recall', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test Loss</div>
            <div class="metric-value">{metrics.get('test_loss', 0):.4f}</div>
        </div>
    </div>
    
    <div class="metric-card">
        <h2>Per-Class Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add per-class metrics
    for class_name, class_metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            html_content += f"""
                <tr>
                    <td><strong>{class_name}</strong></td>
                    <td>{class_metrics.get('precision', 0):.2%}</td>
                    <td>{class_metrics.get('recall', 0):.2%}</td>
                    <td>{class_metrics.get('f1-score', 0):.2%}</td>
                    <td>{class_metrics.get('support', 0)}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="metric-card">
        <h2>Visualizations</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            <div>
                <h3>Confusion Matrix</h3>
                <img src="confusion_matrix.png" alt="Confusion Matrix" style="width: 100%; border-radius: 8px;">
            </div>
            <div>
                <h3>Per-Class Metrics</h3>
                <img src="per_class_metrics.png" alt="Per-Class Metrics" style="width: 100%; border-radius: 8px;">
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Evaluation report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('--metrics-path', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output path for HTML report')
    
    args = parser.parse_args()
    
    generate_html_report(args.metrics_path, args.output_path)


if __name__ == "__main__":
    main()
