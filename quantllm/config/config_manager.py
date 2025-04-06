import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from ..trainer.logger import TrainingLogger

class ConfigManager:
    def __init__(self, logger: Optional[TrainingLogger] = None):
        """
        Initialize the configuration manager.
        
        Args:
            logger (TrainingLogger, optional): Logger instance
        """
        self.logger = logger or TrainingLogger()
        self.configs = {}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            self.logger.log_info(f"Loading configuration from {config_path}")
            
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
            self.configs[config_path.stem] = config
            self.logger.log_info(f"Successfully loaded configuration: {config_path.stem}")
            return config
            
        except Exception as e:
            self.logger.log_error(f"Error loading configuration: {str(e)}")
            raise
            
    def save_config(self, config: Dict[str, Any], config_path: str):
        """
        Save configuration to file.
        
        Args:
            config (Dict[str, Any]): Configuration to save
            config_path (str): Path to save configuration
        """
        try:
            config_path = Path(config_path)
            self.logger.log_info(f"Saving configuration to {config_path}")
            
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif config_path.suffix == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
            self.configs[config_path.stem] = config
            self.logger.log_info(f"Successfully saved configuration: {config_path.stem}")
            
        except Exception as e:
            self.logger.log_error(f"Error saving configuration: {str(e)}")
            raise
            
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name (str): Name of configuration
            
        Returns:
            Dict[str, Any]: Configuration
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration not found: {config_name}")
        return self.configs[config_name]
        
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """
        Update configuration.
        
        Args:
            config_name (str): Name of configuration
            updates (Dict[str, Any]): Updates to apply
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration not found: {config_name}")
            
        try:
            self.logger.log_info(f"Updating configuration: {config_name}")
            self.configs[config_name].update(updates)
            self.logger.log_info(f"Successfully updated configuration: {config_name}")
            
        except Exception as e:
            self.logger.log_error(f"Error updating configuration: {str(e)}")
            raise
            
    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config_name (str): Name of configuration
            schema (Dict[str, Any]): Validation schema
            
        Returns:
            bool: Whether configuration is valid
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration not found: {config_name}")
            
        try:
            self.logger.log_info(f"Validating configuration: {config_name}")
            config = self.configs[config_name]
            
            # Basic schema validation
            for key, value_type in schema.items():
                if key not in config:
                    raise ValueError(f"Missing required key: {key}")
                if not isinstance(config[key], value_type):
                    raise TypeError(f"Invalid type for {key}: expected {value_type}, got {type(config[key])}")
                    
            self.logger.log_info(f"Configuration validation successful: {config_name}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Configuration validation failed: {str(e)}")
            raise 