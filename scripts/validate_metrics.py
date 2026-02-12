"""
Model Metrics Validation Script
Validates that model meets minimum quality thresholds
"""
import argparse
import json
import sys


def validate_metrics(metrics_file, min_accuracy=0.85, min_f1_score=0.80):
    """Validate model metrics against thresholds"""
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    report = data.get('classification_report', {})
    
    # Extract values
    test_accuracy = metrics.get('test_accuracy', 0)
    weighted_avg = report.get('weighted avg', {})
    f1_score = weighted_avg.get('f1-score', 0)
    
    print("=" * 60)
    print("MODEL VALIDATION REPORT")
    print("=" * 60)
    print(f"\nTest Accuracy: {test_accuracy:.4f} (Minimum: {min_accuracy:.4f})")
    print(f"F1-Score: {f1_score:.4f} (Minimum: {min_f1_score:.4f})")
    
    # Validation checks
    passed = True
    
    if test_accuracy < min_accuracy:
        print(f"\n❌ FAILED: Test accuracy {test_accuracy:.4f} is below threshold {min_accuracy:.4f}")
        passed = False
    else:
        print(f"\n✅ PASSED: Test accuracy meets threshold")
    
    if f1_score < min_f1_score:
        print(f"❌ FAILED: F1-score {f1_score:.4f} is below threshold {min_f1_score:.4f}")
        passed = False
    else:
        print(f"✅ PASSED: F1-score meets threshold")
    
    print("\n" + "=" * 60)
    
    if passed:
        print("✅ MODEL VALIDATION PASSED - Ready for deployment")
        print("=" * 60)
        return 0
    else:
        print("❌ MODEL VALIDATION FAILED - Model does not meet quality gates")
        print("=" * 60)
        return 1


def main():
    parser = argparse.ArgumentParser(description='Validate model metrics')
    parser.add_argument('--metrics-file', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--min-accuracy', type=float, default=0.85,
                        help='Minimum required accuracy')
    parser.add_argument('--min-f1-score', type=float, default=0.80,
                        help='Minimum required F1-score')
    
    args = parser.parse_args()
    
    exit_code = validate_metrics(
        args.metrics_file,
        args.min_accuracy,
        args.min_f1_score
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
