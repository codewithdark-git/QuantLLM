import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from ..config import ModelConfig
from tqdm.auto import tqdm
import warnings

class Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._device = self._get_device()
        
        # Configure logging
        logging.getLogger("transformers").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")
        
        self._load_model()
        self._load_tokenizer()

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # For Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        """Load the model with fallback options for different hardware configurations."""
        try:
            kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }

            # Handle quantization settings
            if self._device.type == "cuda":
                if self.config.load_in_4bit:
                    kwargs.update({
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.bfloat16,
                        "bnb_4bit_quant_type": "nf4",
                    })
                elif self.config.load_in_8bit:
                    kwargs.update({"load_in_8bit": True})
            else:
                # CPU or MPS optimizations
                if self.config.cpu_offload:
                    kwargs.update({
                        "device_map": "auto",
                        "offload_folder": "offload",
                        "torch_dtype": torch.float32,
                    })
                else:
                    kwargs.update({
                        "low_cpu_mem_usage": True,
                        "torch_dtype": torch.float32,
                    })

            # Load the model with progress indication via tqdm
            with tqdm(desc="Loading model", total=1, unit="steps") as pbar:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    **kwargs
                )
                pbar.update(1)

            # Apply LoRA if specified and supported
            if self.config.use_lora and self._device.type == "cuda":
                from peft import get_peft_model, LoraConfig
                with tqdm(desc="Applying LoRA configuration", total=1, unit="steps") as pbar:
                    lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05,
                        bias="none",
                    )
                    self.model = get_peft_model(self.model, lora_config)
                    pbar.update(1)
            
            # Move model to appropriate device if not using device_map="auto"
            if "device_map" not in kwargs:
                self.model.to(self._device)

            logging.info(f"Model loaded successfully on {self._device}")

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            # Fallback to CPU with minimal settings if initial loading fails
            logging.info("Attempting fallback to CPU with minimal settings...")
            with tqdm(desc="Loading model (fallback mode)", total=1, unit="steps") as pbar:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    device_map=None,
                )
                pbar.update(1)
            self._device = torch.device("cpu")
            self.model.to(self._device)
            logging.info("Model loaded in fallback mode on CPU")

    def _load_tokenizer(self):
        """Load the tokenizer with appropriate settings."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_model(self):
        """Get the loaded model."""
        return self.model

    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer

    def get_device(self):
        """Get the current device."""
        return self._device