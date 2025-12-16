from huggingface_hub import login, HfApi, Repository
from typing import Optional, Dict, Any
from ..utils import logger, print_success, print_header, print_info, print_error, QuantLLMProgress
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
        print_header("Saving Model for Hub")
        
        save_path = self.staging_dir
        
        # Handle TurboModel wrapper
        if hasattr(model, "model"):
            model_obj = model.model
            tokenizer_obj = getattr(model, "tokenizer", tokenizer)
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
            
        # Generate Model Card with hyperparameters
        self._generate_model_card()
        
        print_success("Model saved to staging area")

    def _generate_model_card(self):
        """Generate a proper README.md for the model card."""
        readme_path = os.path.join(self.staging_dir, "README.md")
        
        content = [
            "---",
            f"base_model: {self.hyperparameters.get('base_model', 'unknown')}",
            "library_name: quantllm",
            "tags:",
            "  - quantllm",
            "  - generated_from_trainer",
            "---",
            "",
            f"# {self.repo_id.split('/')[-1]}",
            "",
            "This model was fine-tuned using [QuantLLM](https://github.com/codewithdark-git/QuantLLM).",
            "",
            "## Hyperparameters",
            "The following hyperparameters were used during training:",
            ""
        ]
        
        for k, v in self.hyperparameters.items():
            content.append(f"- **{k}**: {v}")
            
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def push(self, commit_message: str = "Upload model via QuantLLM"):
        """Push the staged model to HuggingFace Hub."""
        # Disable HF bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        
        try:
            print_header(f"Pushing to {self.repo_id}")
            
            # Ensure repo exists
            if not self.api.repo_exists(self.repo_id):
                print_info(f"Creating repository {self.repo_id}")
                self.api.create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    exist_ok=True
                )

            with QuantLLMProgress() as p:
                task = p.add_task("Uploading artifacts...", total=None)
                
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.staging_dir,
                    commit_message=commit_message,
                    token=self.token
                )
                
                p.update(task, completed=100)
            
            print_success(f"Successfully pushed to https://huggingface.co/{self.repo_id}")
            
            # Clean up staging? Maybe keep it for safety.
            
        except Exception as e:
            print_error(f"Error pushing to Hub: {str(e)}")
            raise

    # Legacy compatibility (optional)
    def push_model(self, *args, **kwargs):
        """Legacy method alias."""
        print_info("Using legacy push_model compatibility")
        pass # Not implementing full legacy logic unless needed
