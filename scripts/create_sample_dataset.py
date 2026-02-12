"""
Create Sample Test Dataset for CI/CD
This script creates a small test dataset from the original dataset
for use in CI/CD pipeline evaluation.
"""
import os
import shutil
import random
from pathlib import Path


def create_sample_dataset(source_dir, target_dir, samples_per_class=5):
    """Create a small sample dataset for testing"""
    
    print(f"Creating sample test dataset...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Samples per class: {samples_per_class}")
    
    # Get all class directories
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"❌ No class directories found in {source_dir}")
        return False
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n📁 Processing class: {class_name}")
        
        # Get all images in this class
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        if not images:
            print(f"  ⚠️  No images found in {class_name}")
            continue
        
        # Randomly sample images
        sample_images = random.sample(images, min(samples_per_class, len(images)))
        
        # Create target class directory
        target_class_dir = target_path / class_name
        target_class_dir.mkdir(exist_ok=True)
        
        # Copy sampled images
        for img in sample_images:
            target_img = target_class_dir / img.name
            shutil.copy2(img, target_img)
            total_copied += 1
        
        print(f"  ✓ Copied {len(sample_images)} images")
    
    print(f"\n✓ Sample dataset created successfully!")
    print(f"  Total images: {total_copied}")
    print(f"  Location: {target_dir}")
    
    return True


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    source_dir = "Data/OriginalDataset"
    target_dir = "Data/SampleTestDataset"
    
    # Create sample dataset
    success = create_sample_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        samples_per_class=5
    )
    
    if success:
        print("\n✓ Done! You can now commit this sample dataset to Git.")
        print(f"  git add {target_dir}")
        print(f"  git commit -m 'Add sample test dataset for CI/CD'")
    else:
        print("\n❌ Failed to create sample dataset")
        exit(1)


if __name__ == "__main__":
    main()
