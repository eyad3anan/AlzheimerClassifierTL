"""
Model Training Script with MLflow Tracking
"""
import os
import argparse
import json
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def create_model(num_classes=4, learning_rate=0.001):
    """Create ResNet50 transfer learning model"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def prepare_data(data_dir, batch_size=32, img_size=(224, 224)):
    """Prepare data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator


def main():
    parser = argparse.ArgumentParser(description='Train Alzheimer Classifier')
    parser.add_argument('--data-dir', type=str, default='Data/OriginalDataset/train',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mlflow-tracking-uri', type=str, default='http://localhost:5000',
                        help='MLflow tracking URI')
    parser.add_argument('--experiment-name', type=str, default='alzheimer-classifier',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Try to set up MLflow tracking (optional)
    mlflow_available = False
    try:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        mlflow_available = True
        print(f"✓ MLflow tracking enabled at {args.mlflow_tracking_uri}")
    except Exception as e:
        print(f"⚠ MLflow tracking unavailable: {e}")
        print("Continuing training without MLflow tracking...")
    
    # Training function
    def train_model():
        # Log parameters to MLflow if available
        if mlflow_available:
            try:
                mlflow.log_param("epochs", args.epochs)
                mlflow.log_param("batch_size", args.batch_size)
                mlflow.log_param("learning_rate", args.learning_rate)
                mlflow.log_param("model_architecture", "ResNet50")
            except Exception as e:
                print(f"⚠ Could not log parameters to MLflow: {e}")
        
        # Prepare data
        print("Preparing data...")
        train_gen, val_gen = prepare_data(args.data_dir, args.batch_size)
        
        # Create model
        print("Creating model...")
        model = create_model(num_classes=4, learning_rate=args.learning_rate)
        
        # Log model summary
        model.summary()
        
        # Callbacks
        os.makedirs('models', exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ]
        
        # Train model
        print("Training model...")
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log metrics to MLflow if available
        if mlflow_available:
            try:
                for epoch in range(len(history.history['loss'])):
                    mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                    mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            except Exception as e:
                print(f"⚠ Could not log metrics to MLflow: {e}")
        
        # Save final model
        model.save('models/AlzheimerClassifierTL.h5')
        
        # Log model to MLflow if available
        if mlflow_available:
            try:
                mlflow.tensorflow.log_model(model, "model")
            except Exception as e:
                print(f"⚠ Could not log model to MLflow: {e}")
        
        # Save training history
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/training_history.json', 'w') as f:
            json.dump(history.history, f)
        
        # Log artifact to MLflow if available
        if mlflow_available:
            try:
                mlflow.log_artifact('outputs/training_history.json')
            except Exception as e:
                print(f"⚠ Could not log artifact to MLflow: {e}")
        
        print(f"\n✓ Training completed! Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"✓ Model saved to models/AlzheimerClassifierTL.h5")
        return history
    
    # Run training with or without MLflow
    if mlflow_available:
        try:
            with mlflow.start_run():
                train_model()
        except Exception as e:
            print(f"⚠ MLflow run failed: {e}")
            print("Running training without MLflow...")
            train_model()
    else:
        train_model()


if __name__ == "__main__":
    main()
