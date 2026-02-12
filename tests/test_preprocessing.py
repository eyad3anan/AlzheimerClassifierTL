"""
Unit tests for preprocessing and model utilities
"""
import pytest
import numpy as np
from PIL import Image
import tensorflow as tf


class TestPreprocessing:
    """Test preprocessing functions"""
    
    def test_image_resize(self):
        """Test image resizing to 224x224"""
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # Resize
        resized = test_image.resize((224, 224))
        
        assert resized.size == (224, 224)
    
    def test_image_to_array(self):
        """Test image to array conversion"""
        test_image = Image.new('RGB', (224, 224), color='blue')
        img_array = np.array(test_image)
        
        assert img_array.shape == (224, 224, 3)
        assert img_array.dtype == np.uint8
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        test_image = Image.new('RGB', (512, 512), color='green')
        
        # Preprocess
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((224, 224))
        img_array = np.array(test_image)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        assert img_array.shape == (1, 224, 224, 3)
        assert img_array.dtype == np.float32


class TestModelStructure:
    """Test model structure and configuration"""
    
    def test_class_mapping(self):
        """Test class mapping dictionary"""
        class_mapping = {
            "MildDemented": 0,
            "ModerateDemented": 1,
            "NonDemented": 2,
            "VeryMildDemented": 3
        }
        
        assert len(class_mapping) == 4
        assert "NonDemented" in class_mapping
        assert class_mapping["NonDemented"] == 2
    
    def test_reverse_mapping(self):
        """Test reverse class mapping"""
        class_mapping = {
            "MildDemented": 0,
            "ModerateDemented": 1,
            "NonDemented": 2,
            "VeryMildDemented": 3
        }
        idx_to_class = {v: k for k, v in class_mapping.items()}
        
        assert len(idx_to_class) == 4
        assert idx_to_class[2] == "NonDemented"


class TestStatusMessages:
    """Test status message configuration"""
    
    def test_status_messages_exist(self):
        """Test that all status messages are defined"""
        status_messages = {
            "NonDemented": ("✅ Likely Healthy / No Dementia detected", "green"),
            "VeryMildDemented": ("🟡 Signs of Very Mild Dementia", "orange"),
            "MildDemented": ("🟠 Signs of Mild Dementia", "darkorange"),
            "ModerateDemented": ("🔴 Signs of Moderate Dementia", "darkred")
        }
        
        assert len(status_messages) == 4
        assert all(isinstance(v, tuple) and len(v) == 2 for v in status_messages.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
