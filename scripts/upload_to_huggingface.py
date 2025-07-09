#!/usr/bin/env python3
"""
Upload the Foreign Affairs dataset to HuggingFace Hub as a private dataset.

This script:
1. Authenticates with HuggingFace
2. Creates a private dataset repository
3. Uploads the dataset files
4. Sets appropriate metadata
"""

import os
import sys
from pathlib import Path
from typing import Optional
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from datasets import load_from_disk
except ImportError:
    print("Error: Please install required libraries")
    print("Run: pip install huggingface_hub datasets")
    sys.exit(1)


class HuggingFaceUploader:
    """Upload dataset to HuggingFace Hub"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize uploader with HuggingFace token.
        
        Args:
            token: HuggingFace API token. If None, will try to use HF_TOKEN env var
        """
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "No HuggingFace token found. Please set HF_TOKEN environment variable "
                "or pass token parameter. Get your token from: https://huggingface.co/settings/tokens"
            )
        
        self.api = HfApi(token=self.token)
    
    def create_private_repo(self, repo_name: str, organization: Optional[str] = None) -> str:
        """
        Create a private dataset repository on HuggingFace.
        
        Args:
            repo_name: Name of the dataset repository
            organization: Optional organization name (uses your username if None)
            
        Returns:
            Full repository ID
        """
        # Get current user info
        user_info = self.api.whoami()
        username = user_info["name"]
        
        # Construct full repo ID
        if organization:
            repo_id = f"{organization}/{repo_name}"
        else:
            repo_id = f"{username}/{repo_name}"
        
        print(f"Creating private repository: {repo_id}")
        
        try:
            # Create the repository
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=True,
                token=self.token,
                exist_ok=True
            )
            print(f"✅ Repository created/verified: {repo_id}")
            
        except Exception as e:
            print(f"❌ Error creating repository: {e}")
            raise
        
        return repo_id
    
    def upload_dataset(self, dataset_path: str, repo_id: str, commit_message: str = None):
        """
        Upload dataset files to HuggingFace repository.
        
        Args:
            dataset_path: Local path to the dataset directory
            repo_id: HuggingFace repository ID
            commit_message: Optional commit message
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        # Default commit message
        if not commit_message:
            commit_message = "Upload Foreign Affairs dataset"
        
        print(f"\n📤 Uploading dataset from {dataset_path} to {repo_id}...")
        
        try:
            # Upload the entire folder
            upload_folder(
                folder_path=str(dataset_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message,
                ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"]
            )
            
            print(f"✅ Dataset uploaded successfully!")
            print(f"🔗 View your dataset at: https://huggingface.co/datasets/{repo_id}")
            
        except Exception as e:
            print(f"❌ Error uploading dataset: {e}")
            raise
    
    def verify_upload(self, repo_id: str):
        """Verify the dataset was uploaded correctly"""
        try:
            # List files in the repository
            files = self.api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            print(f"\n📁 Files in repository:")
            for file in sorted(files):
                print(f"  • {file}")
            
            # Check for essential files
            essential_files = ["README.md", "dataset_stats.json"]
            parquet_files = [f for f in files if f.endswith('.parquet')]
            
            if all(f in files for f in essential_files) and parquet_files:
                print(f"\n✅ Upload verified - all essential files present")
                return True
            else:
                print(f"\n⚠️  Warning: Some essential files may be missing")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying upload: {e}")
            return False


def main():
    """Main upload function"""
    print("🚀 HuggingFace Dataset Uploader")
    print("=" * 60)
    
    # Configuration
    dataset_path = "/Users/jsv/Projects/foreign_affairs/rem_rag_v2/data/huggingface_dataset"
    repo_name = "foreign_affairs_essays_1922_2024"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please run create_huggingface_dataset.py first")
        return
    
    # Load dataset stats for info
    stats_path = Path(dataset_path) / "dataset_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"\n📊 Dataset info:")
        print(f"  • Total articles: {stats['total_articles']:,}")
        print(f"  • Year range: {stats['year_range']['min']}-{stats['year_range']['max']}")
    
    # Check for HuggingFace token
    if not os.environ.get("HF_TOKEN"):
        print("\n⚠️  No HuggingFace token found!")
        print("Please set your token:")
        print("  export HF_TOKEN='your_token_here'")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        print("Make sure it has 'write' permissions")
        return
    
    # Confirm upload
    print(f"\n🔐 This will create a PRIVATE dataset repository")
    print(f"Repository name: {repo_name}")
    response = input("\nProceed with upload? (y/n): ")
    
    if response.lower() != 'y':
        print("Upload cancelled")
        return
    
    # Initialize uploader
    try:
        uploader = HuggingFaceUploader()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Get user info
    user_info = uploader.api.whoami()
    username = user_info["name"]
    print(f"\n👤 Logged in as: {username}")
    
    # Create repository
    try:
        repo_id = uploader.create_private_repo(repo_name)
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return
    
    # Upload dataset
    try:
        uploader.upload_dataset(
            dataset_path=dataset_path,
            repo_id=repo_id,
            commit_message="Initial upload of Foreign Affairs essays (1922-2024)"
        )
    except Exception as e:
        print(f"❌ Failed to upload dataset: {e}")
        return
    
    # Verify upload
    uploader.verify_upload(repo_id)
    
    print("\n✨ Upload complete!")
    print(f"\n📝 Next steps:")
    print(f"1. Visit your dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"2. Verify the dataset card looks correct")
    print(f"3. Test loading the dataset with:")
    print(f"   from datasets import load_dataset")
    print(f"   ds = load_dataset('{repo_id}', use_auth_token=True)")
    
    # Save upload info
    from datetime import datetime
    upload_info = {
        "repo_id": repo_id,
        "username": username,
        "dataset_path": str(dataset_path),
        "upload_date": datetime.now().isoformat(),
        "private": True
    }
    
    info_path = Path(dataset_path) / "upload_info.json"
    with open(info_path, 'w') as f:
        json.dump(upload_info, f, indent=2)
    print(f"\n💾 Upload info saved to: {info_path}")


if __name__ == "__main__":
    main()