"""Factory for creating the appropriate assistant based on model type."""
from typing import Dict, Any, Literal
from src.models.base import BaseAssistant
from src.models.muse_glimmer_assistant import MuseGlimmerAssistant
from src.models.openai_assistant import OpenAIAssistant
from src.models.openweight_assistant import OpenWeightAssistant
from src.models.qwen35_assistant import HybridQwenAssistant
from src.models.gemma4_assistant import Gemma4Assistant
from src.models.registry import MODEL_REGISTRY
from src.models.saul_assistant import SaulAssistant



class AssistantFactory:
    """Factory for creating assistant instances.
    
    Usage:
        # Create OpenAI assistant
        assistant = AssistantFactory.create("openai", model_name="gpt-4")
        
        # Create open-weight assistant
        assistant = AssistantFactory.create("openweight", model_name="Qwen/Qwen2.5-8B")
        
        # Use the assistant
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
        response = assistant.generate(messages)
    """
    
    # Mapping of model identifiers to assistant types
    MODEL_TYPES = {
        # OpenAI models
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4o": "openai",
        
        # Open-weight models
        "qwen": "openweight",
        "qwen2.5": "openweight",
        "phi": "openweight",
        "saullm": "openweight",
        "mistral": "openweight",
        "llama": "openweight",
        # Encoders (token-classification, see src.training.infer_encoder)
        "deberta": "encoder",
        "modernbert": "encoder",
    }
    
    @staticmethod
    def create(
        model_name: str,
        temperature: float = 0.3,
        quantization: str = None,
        **kwargs
    ) -> BaseAssistant:
        """Create an assistant instance.
        
        Args:
            model_name: Name or path of the model
            temperature: Sampling temperature (0.0 to 2.0)
            quantization: Quantization method (optional)
            **kwargs: Additional model-specific parameters
        
        Returns:
            BaseAssistant: An instance of the appropriate assistant
        
        Raises:
            ValueError: If assistant_type is not supported
        """
        if kwargs.get("assistant_type") == "openai":
            return OpenAIAssistant(model_name=model_name, temperature=temperature, **kwargs)
        
        if model_name not in MODEL_REGISTRY:
            # Case-insensitive fallback (e.g. "modernbert-large" vs "ModernBERT-large")
            lowered = {k.lower(): k for k in MODEL_REGISTRY}
            if model_name.lower() in lowered:
                model_name = lowered[model_name.lower()]
            else:
                raise ValueError(
                    f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
                )
        cfg = MODEL_REGISTRY[model_name]
        quant = quantization or cfg["default_quantization"]

        if cfg["type"] == "openweight":
            return OpenWeightAssistant(
                model_name=model_name,
                model_path=cfg["path"],
                temperature=temperature,
                quantization=quant,
                trust_remote_code=cfg["trust_remote_code"],
                has_system_role=cfg["has_system_role"],
                thinking=cfg["thinking"],
                **kwargs,
            )

        elif cfg["type"] == "hybrid":
            return HybridQwenAssistant(
                model_name=model_name,
                model_path=cfg["path"],
                temperature=temperature,
                quantization=quant,
                trust_remote_code=cfg["trust_remote_code"],
                has_system_role=cfg["has_system_role"],
                thinking=cfg["thinking"],
                **kwargs,
            )

        elif cfg["type"] == "multimodal":
            return MuseGlimmerAssistant(
                model_name=model_name,
                model_path=cfg["path"],
                temperature=temperature,
                quantization=quant,
                trust_remote_code=cfg["trust_remote_code"],
                has_system_role=cfg["has_system_role"],
                thinking=cfg["thinking"],
                **kwargs,
            )
        
        elif cfg["type"] == "gemma4":
            return Gemma4Assistant(
                model_name=model_name,
                model_path=cfg["path"],
                temperature=temperature,
                quantization=quant,
                trust_remote_code=cfg["trust_remote_code"],
                has_system_role=cfg["has_system_role"],
                thinking=cfg["thinking"],
                **kwargs,
            )

        elif cfg["type"] == "saul":
            return SaulAssistant(
                model_name=model_name,
                model_path=cfg["path"],
                temperature=temperature,
                quantization=quant,
                trust_remote_code=cfg["trust_remote_code"],
                has_system_role=cfg["has_system_role"],
                thinking=cfg["thinking"],
                **kwargs,
            )

        elif cfg["type"] == "encoder":
            from src.training.infer_encoder import EncoderAssistant

            adapter_path = kwargs.pop("adapter_path", None)
            return EncoderAssistant(
                adapter_path=adapter_path,
                temperature=temperature,
                has_system_role=cfg["has_system_role"],
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown type: {cfg['type']}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseAssistant:
        """Create an assistant from a configuration dictionary.
        
        Args:
            config: Dictionary containing:
                - "type": "openai" or "openweight"
                - "model_name": Model name or path
                - "temperature": (optional) Sampling temperature
                - Other model-specific parameters
        
        Returns:
            BaseAssistant: An instance of the appropriate assistant
        
        Example:
            config = {
                "type": "openweight",
                "model_name": "Qwen/Qwen2.5-8B",
                "temperature": 0.5,
                "max_tokens": 1024,
                "device": "cuda"
            }
            assistant = AssistantFactory.create_from_config(config)
        """
        config = config.copy()
        assistant_type = config.pop("type", None)
        model_name = config.pop("model_name", None)
        temperature = config.pop("temperature", 0.3)
        
        if not assistant_type:
            raise ValueError("Config must include 'type' key")
        if not model_name:
            raise ValueError("Config must include 'model_name' key")
        
        return AssistantFactory.create(
            assistant_type=assistant_type,
            model_name=model_name,
            temperature=temperature,
            **config
        )
    
    @staticmethod
    def auto_detect(model_name: str, **kwargs) -> BaseAssistant:
        """Auto-detect model type and create appropriate assistant.
        
        Args:
            model_name: Name or path of the model
            **kwargs: Additional parameters (temperature, device, etc.)
        
        Returns:
            BaseAssistant: An instance of the appropriate assistant
        
        Note:
            This uses heuristics to detect model type. Explicit creation is recommended.
        """
        # Convert to lowercase for matching
        model_lower = model_name.lower()
        
        # Check against known patterns
        for pattern, assistant_type in AssistantFactory.MODEL_TYPES.items():
            if pattern in model_lower:
                return AssistantFactory.create(
                    assistant_type=assistant_type,
                    model_name=model_name,
                    **kwargs
                )
        
        # Default to openweight for unknown models
        print(f"Warning: Unknown model {model_name}, assuming open-weight model")
        return AssistantFactory.create(
            assistant_type="openweight",
            model_name=model_name,
            **kwargs
        )
