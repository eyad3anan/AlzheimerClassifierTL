"""
Upload Model to Hugging Face Hub
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo


def upload_to_huggingface(model_path, repo_id, token, commit_message="Update model"):
    """Upload model to Hugging Face Hub"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Uploading model to Hugging Face Hub: {repo_id}")
    
    # Initialize API
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Repository already exists or error: {e}")
    
    # Upload model file
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        print(f"✅ Model uploaded successfully to {repo_id}")
        print(f"Model URL: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model file')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='Hugging Face repository ID (username/repo-name)')
    parser.add_argument('--token', type=str, required=True,
                        help='Hugging Face API token')
    parser.add_argument('--commit-message', type=str, 
                        default='Update model via CI/CD',
                        help='Commit message')
    
    args = parser.parse_args()
    
    upload_to_huggingface(
        args.model_path,
        args.repo_id,
        args.token,
        args.commit_message
    )


if __name__ == "__main__":
    main()
