from huggingface_hub import HfApi, Repository
from typing import Optional, Dict, Any
import os
from ..finetune.logger import TrainingLogger

class HubManager:
    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize the HuggingFace Hub manager.
        
        Args:
            model_id (str): Model ID on HuggingFace Hub
            token (str, optional): HuggingFace API token
            organization (str, optional): Organization name
        """
        self.model_id = model_id
        self.token = token
        self.organization = organization
        self.api = HfApi()
        self.logger = TrainingLogger()
        
    def push_model(
        self,
        model,
        tokenizer,
        commit_message: str = "Update model",
        **kwargs
    ):
        """
        Push model and tokenizer to HuggingFace Hub.
        
        Args:
            model: The model to push
            tokenizer: The tokenizer to push
            commit_message (str): Commit message
            **kwargs: Additional arguments for push
        """
        try:
            # Create repository if it doesn't exist
            if not self.api.repo_exists(self.model_id):
                self.api.create_repo(
                    repo_id=self.model_id,
                    token=self.token,
                    organization=self.organization
                )
                
            # Push model
            model.push_to_hub(
                self.model_id,
                token=self.token,
                commit_message=commit_message,
                **kwargs
            )
            
            # Push tokenizer
            tokenizer.push_to_hub(
                self.model_id,
                token=self.token,
                commit_message=commit_message,
                **kwargs
            )
            
            self.logger.log_info(f"Successfully pushed model to {self.model_id}")
            
        except Exception as e:
            self.logger.log_error(f"Error pushing model: {str(e)}")
            raise
            
    def push_checkpoint(
        self,
        checkpoint_path: str,
        commit_message: str = "Update checkpoint",
        **kwargs
    ):
        """
        Push checkpoint to HuggingFace Hub.
        
        Args:
            checkpoint_path (str): Path to checkpoint
            commit_message (str): Commit message
            **kwargs: Additional arguments for push
        """
        try:
            # Create repository if it doesn't exist
            if not self.api.repo_exists(self.model_id):
                self.api.create_repo(
                    repo_id=self.model_id,
                    token=self.token,
                    organization=self.organization
                )
                
            # Push checkpoint
            self.api.upload_folder(
                folder_path=checkpoint_path,
                repo_id=self.model_id,
                token=self.token,
                commit_message=commit_message,
                **kwargs
            )
            
            self.logger.log_info(f"Successfully pushed checkpoint to {self.model_id}")
            
        except Exception as e:
            self.logger.log_error(f"Error pushing checkpoint: {str(e)}")
            raise
            
    def pull_model(self, local_dir: str = None):
        """
        Pull model from HuggingFace Hub.
        
        Args:
            local_dir (str, optional): Local directory to save model
        """
        try:
            if local_dir is None:
                local_dir = self.model_id.split("/")[-1]
                
            # Clone repository
            repo = Repository(
                local_dir=local_dir,
                clone_from=self.model_id,
                token=self.token
            )
            
            self.logger.log_info(f"Successfully pulled model to {local_dir}")
            return local_dir
            
        except Exception as e:
            self.logger.log_error(f"Error pulling model: {str(e)}")
            raise 