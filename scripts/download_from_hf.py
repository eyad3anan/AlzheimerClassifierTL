"""
Download Model from Hugging Face Hub
"""
import argparse
import os
from huggingface_hub import hf_hub_download


def download_model(repo_id, filename, token, output_dir="models"):
    """Download model from Hugging Face Hub"""
    try:
        print(f"📥 Downloading model from {repo_id}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the model
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            cache_dir=".cache",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Model downloaded successfully to: {model_path}")
        
        # Also copy to expected location
        expected_path = os.path.join(output_dir, filename)
        if model_path != expected_path and not os.path.exists(expected_path):
            import shutil
            shutil.copy2(model_path, expected_path)
            print(f"✓ Model copied to: {expected_path}")
        
        return expected_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Download model from Hugging Face')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='Hugging Face repository ID')
    parser.add_argument('--filename', type=str, default='AlzheimerClassifierTL.h5',
                        help='Model filename')
    parser.add_argument('--token', type=str, required=True,
                        help='Hugging Face API token')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory')
    
    args = parser.parse_args()
    
    model_path = download_model(
        repo_id=args.repo_id,
        filename=args.filename,
        token=args.token,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Model ready at: {model_path}")


if __name__ == "__main__":
    main()
