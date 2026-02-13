"""
Model Evaluation Script
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_path, test_data_path, batch_size=32):
    """Evaluate model on test data"""
    # Load model with safe_mode=False to handle DTypePolicy compatibility
    print(f"Loading model from {model_path}")
    
    try:
        # Load with safe_mode=False to skip strict validation
        model = load_model(model_path, compile=False, safe_mode=False)
        
        # Recompile with standard metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model loaded successfully (safe_mode=False)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}. The model may be incompatible with the current Keras version.")
    
    # Prepare test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    print("Evaluating model...")
    results = model.evaluate(test_generator, verbose=1)
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics from sklearn
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {
        'test_loss': float(results[0]) if isinstance(results, list) else float(results),
        'test_accuracy': float(results[1]) if isinstance(results, list) and len(results) > 1 else float(results),
        'test_precision': float(precision_score(y_true, y_pred, average='weighted')),
        'test_recall': float(recall_score(y_true, y_pred, average='weighted')),
        'test_f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    # Classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save metrics
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'classification_report': report
        }, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    axes[0].bar(classes, precision, color='skyblue')
    axes[0].set_title('Precision by Class')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(classes, recall, color='lightgreen')
    axes[1].set_title('Recall by Class')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(classes, f1, color='lightcoral')
    axes[2].set_title('F1-Score by Class')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print("\nMetrics saved to outputs/metrics.json")
    print("Visualizations saved to outputs/")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Alzheimer Classifier')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--test-data-path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data_path, args.batch_size)


if __name__ == "__main__":
    main()
