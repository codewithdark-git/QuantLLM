from huggingface_hub import HfApi
from typing import Optional, Dict, Any, List
from ..utils import (
    logger, 
    print_success, 
    print_header, 
    print_info, 
    print_error, 
    print_warning,
    QuantLLMProgress,
    format_size,
)
from .model_card import ModelCardGenerator, generate_model_card
import os

class QuantLLMHubManager:
    def __init__(
        self,
        repo_id: str,
        hf_token: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize HubManager for lifecycle management.
        
        Args:
           repo_id: HuggingFace repository ID (e.g., "username/model")
           hf_token: HuggingFace API token
           organization: Optional organization name
        """
        self.repo_id = repo_id
        self.organization = organization
        self.token = hf_token
        self.api = HfApi(token=hf_token)
        
        # State
        self.hyperparameters = {}
        self.staging_dir = f"./hub_staging/{repo_id.split('/')[-1]}"
        self._ensure_staging_dir()
        
        # Login if token provided
        if hf_token:
            self.login()

    def _ensure_staging_dir(self):
        if not os.path.exists(self.staging_dir):
            os.makedirs(self.staging_dir)

    def login(self):
        """Login to Hugging Face Hub."""
        try:
            self.api = HfApi(token=self.token)
            # Verify login by getting user info
            user = self.api.whoami(self.token)
            print_success(f"Successfully logged in as {user['name']}")
        except Exception as e:
            print_error(f"Error logging in: {str(e)}")
            # Don't raise, allows offline usage until push

    def track_hyperparameters(self, params: Dict[str, Any]):
        """
        Track training hyperparameters for the model card.
        
        Args:
            params: Dictionary of hyperparameters
        """
        self.hyperparameters.update(params)
        print_info(f"Tracked {len(params)} hyperparameters")

    def save_final_model(self, model, tokenizer=None, format: str = "safetensors"):
        """
        Save the model and tokenizer to the staging directory.
        
        Args:
            model: The model object (TurboModel or generic PEFT/TF model)
            tokenizer: The tokenizer object
            format: 'safetensors' or 'pytorch'
        """
        print_header("Saving Model for Hub", icon="💾")
        
        save_path = self.staging_dir
        
        # Handle TurboModel wrapper
        if hasattr(model, "model"):
            model_obj = model.model
            tokenizer_obj = getattr(model, "tokenizer", tokenizer)
            # Get base model name
            if hasattr(model_obj, 'config') and hasattr(model_obj.config, '_name_or_path'):
                base_model = model_obj.config._name_or_path
                self.hyperparameters['base_model'] = base_model
        else:
            model_obj = model
            tokenizer_obj = tokenizer
            
        # Save model
        print_info(f"Saving model to {save_path}...")
        safe_serialization = (format == "safetensors")
        
        model_obj.save_pretrained(
            save_path, 
            safe_serialization=safe_serialization
        )
        
        # Save tokenizer if available
        if tokenizer_obj:
            print_info("Saving tokenizer...")
            tokenizer_obj.save_pretrained(save_path)
        
        print_success("Model saved to staging area")

    def _generate_model_card(self, format: str = "safetensors"):
        """
        Generate a proper README.md with format-specific usage examples.
        
        Args:
            format: Export format (gguf, onnx, mlx, safetensors)
        """
        readme_path = os.path.join(self.staging_dir, "README.md")
        
        # Get parameters
        base_model = self.hyperparameters.get('base_model', 'unknown')
        quantization = self.hyperparameters.get('quantization')
        license_type = self.hyperparameters.get('license', 'apache-2.0')
        
        # Collect tags
        extra_tags = []
        if self.hyperparameters.get('finetuned'):
            extra_tags.append("finetuned")
        
        # Generate model card
        content = generate_model_card(
            repo_id=self.repo_id,
            base_model=base_model,
            format=format,
            quantization=quantization,
            license=license_type,
            tags=extra_tags,
        )
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print_info(f"Generated model card for {format.upper()} format")

    def push(self, commit_message: str = "Upload model via QuantLLM"):
        """Push the staged model to HuggingFace Hub."""
        # Disable HF bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        
        try:
            print_header(f"Pushing to {self.repo_id}", icon="📤")
            
            # Ensure repo exists
            if not self.api.repo_exists(self.repo_id):
                print_info(f"Creating repository {self.repo_id}")
                self.api.create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    exist_ok=True
                )

            # Calculate total size
            total_size = 0
            for root, dirs, files in os.walk(self.staging_dir):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
            
            print_info(f"Uploading {format_size(total_size)}...")

            with QuantLLMProgress() as p:
                task = p.add_task("Uploading to HuggingFace Hub...", total=None)
                
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.staging_dir,
                    commit_message=commit_message,
                    token=self.token
                )
                
                p.update(task, completed=100)
            
            print_success(f"Pushed to https://huggingface.co/{self.repo_id}")
            
        except Exception as e:
            print_error(f"Error pushing to Hub: {str(e)}")
            raise

    # Legacy compatibility (optional)
    def push_model(self, *args, **kwargs):
        """Legacy method alias."""
        print_info("Using legacy push_model compatibility")
        pass # Not implementing full legacy logic unless needed
