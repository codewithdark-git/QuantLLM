from huggingface_hub import login, HfApi, Repository
from typing import Optional, Dict, Any
import os

class HubManager:
    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """Initialize HubManager and login to Hugging Face."""
        self.model_id = model_id
        self.organization = organization
        self.token = token
        
    def login(self):
        """Login to Hugging Face Hub."""
        try:
            self.api = HfApi(token=self.token)
            print("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"Error logging in: {str(e)}")
            raise
                
    def push_model(
        self,
        model,
        tokenizer,
        commit_message: str = "Update model",
        **kwargs
    ):
        """Push model and tokenizer to HuggingFace Hub."""
        try:
            if not self.api.repo_exists(self.model_id):
                self.api.create_repo(
                    repo_id=self.model_id,
                    token=self.token,
                    organization=self.organization
                )
                print(f"Created new repository: {self.model_id}")
                
            # Push model
            model.push_to_hub(
                self.model_id,
                token=self.token,
                commit_message=commit_message,
                **kwargs
            )
            print(f"Successfully pushed model to {self.model_id}")
            
            # Push tokenizer
            tokenizer.push_to_hub(
                self.model_id,
                token=self.token,
                commit_message=commit_message,
                **kwargs
            )
            print(f"Successfully pushed tokenizer to {self.model_id}")
            
        except Exception as e:
            print(f"Error pushing to hub: {str(e)}")
            raise
            
    def push_checkpoint(
        self,
        checkpoint_path: str,
        commit_message: str = "Update checkpoint",
        **kwargs
    ):
        """Push checkpoint to HuggingFace Hub."""
        try:
            if not self.api.repo_exists(self.model_id):
                self.api.create_repo(
                    repo_id=self.model_id,
                    token=self.token,
                    organization=self.organization
                )
                print(f"Created new repository: {self.model_id}")
                
            # Push checkpoint
            self.api.upload_folder(
                folder_path=checkpoint_path,
                repo_id=self.model_id,
                token=self.token,
                commit_message=commit_message,
                **kwargs
            )
            print(f"Successfully pushed checkpoint to {self.model_id}")
            
        except Exception as e:
            print(f"Error pushing checkpoint: {str(e)}")
            raise
            
    def pull_model(self, local_dir: str = None):
        """Pull model from HuggingFace Hub."""
        try:
            if local_dir is None:
                local_dir = self.model_id.split("/")[-1]
                
            # Clone repository
            repo = Repository(
                local_dir=local_dir,
                clone_from=self.model_id,
                token=self.token
            )
            print(f"Successfully pulled model to {local_dir}")
            return local_dir
            
        except Exception as e:
            print(f"Error pulling model: {str(e)}")
            raise