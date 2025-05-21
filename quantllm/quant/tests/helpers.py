import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, vocab_size=100, hidden_size=32, input_dim=16, output_dim=10, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Add any other necessary config attributes for PreTrainedModel
        self.num_attention_heads = 4 
        self.num_hidden_layers = 2


class DummyModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self.linear1 = nn.Linear(config.input_dim, config.hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.output_dim)

    def forward(self, input_ids: Optional[torch.Tensor] = None, **kwargs):
        # In a real model, input_ids would be processed. Here, we'll use a dummy input
        # if input_ids is None, or adapt based on input_ids if provided.
        # For simplicity, let's assume input_ids could be our direct input if shaped correctly,
        # or we generate a dummy input matching input_dim.
        
        if input_ids is not None and input_ids.shape[-1] == self.config.input_dim:
            # If input_ids has the correct last dimension, use it directly.
            # This is a simplification; typically input_ids are indices.
            dummy_input = input_ids.float() 
            if dummy_input.ndim == 2: # e.g. (batch_size, input_dim)
                 pass
            elif dummy_input.ndim == 3: # e.g. (batch_size, seq_len, input_dim), take last token's features
                dummy_input = dummy_input[:, -1, :]
            else: # fallback if shape is unexpected
                batch_size = input_ids.shape[0]
                dummy_input = torch.randn(batch_size, self.config.input_dim, device=self.device)

        else: # Fallback or default behavior
            # Determine batch size from input_ids if possible, else default to 1
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            dummy_input = torch.randn(batch_size, self.config.input_dim, device=self.device)

        x = self.linear1(dummy_input)
        x = self.relu(x)
        x = self.linear2(x)
        return x # Simplified output, real models return ModelOutput objects

def create_dummy_model(input_dim=16, hidden_dim=32, output_dim=10) -> PreTrainedModel:
    """
    Creates a simple PreTrainedModel with two Linear layers for testing.
    """
    config = DummyConfig(input_dim=input_dim, hidden_size=hidden_dim, output_dim=output_dim)
    model = DummyModel(config)
    return model

def create_dummy_calibration_data(batch_size: int = 2, vocab_size: int = 1000, seq_len: int = 32, num_samples: int = 8) -> torch.Tensor:
    """
    Creates a random integer tensor representing token IDs for calibration.
    Suitable for language models expecting input_ids.

    Args:
        batch_size (int): Not directly used in current output shape, but common for dataloaders.
                          The output tensor is a single batch of all samples.
        vocab_size (int): The maximum value for token IDs (exclusive).
        seq_len (int): The sequence length of each calibration sample.
        num_samples (int): The number of calibration samples to generate.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, seq_len) with random token IDs.
    """
    # batch_size is not directly used here as we return a single tensor of (num_samples, seq_len)
    # which can then be handled by a DataLoader or processed in chunks by the quantizer.
    return torch.randint(0, vocab_size, (num_samples, seq_len))
