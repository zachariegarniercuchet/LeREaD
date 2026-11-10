# src/models/registry.py
import os

MODELS_DIR = "/home/z/zagar/links/scratch" #"/home/z/zagar/links/scratch" #"/home/zagar/scratch"  # your scratch dir

MODEL_REGISTRY = {
    "Qwen2.5-7B-Instruct": {
        "path": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "Qwen2.5-32B-Instruct": {
        "path": os.path.join(MODELS_DIR, "Qwen2.5-32B-Instruct"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "4bit",  # 32B needs quantization
    },
    "Qwen3.5-9B": {
        "path": os.path.join(MODELS_DIR, "Qwen3.5-9B"),
        "type": "hybrid",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "bf16",  # 3.5B models are released in bf16
    },
    "Qwen3.8-27B": {
        "path": os.path.join(MODELS_DIR, "Qwen3.8-27B"),  # match your actual folder name
        "type": "hybrid",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,          # we disable reasoning
        "default_quantization": "4bit", 
    },
    "SaulLM-7B-Instruct": {
        "path": os.path.join(MODELS_DIR, "SaulLM-7B-Instruct"),
        "type": "saul",
        "has_system_role": False,
        "trust_remote_code": False,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "SaulLM-54B-Instruct": {
        "path": os.path.join(MODELS_DIR, "SaulLm-54B-Instruct"),
        "type": "saul",
        "has_system_role": False,
        "trust_remote_code": False,
        "thinking": False,
        "default_quantization": "8bit", #4bit or fp16
    },
    "phi-4": {
        "path": os.path.join(MODELS_DIR, "phi-4"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "Mistral-7B-Instruct-v0.1": {
        "path": os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.1"),
        "type": "openweight",
        "has_system_role": False,  # older Mistral has no system role
        "trust_remote_code": False,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "Muse-Glimmer-30B": {
        "path": os.path.join(MODELS_DIR, "Muse-Glimmer-30B"),
        "type": "multimodal",
        "has_system_role": True,
        "trust_remote_code": True,   # brand-new arch (muse_glimmer); need a transformers
                                      # build that ships it, upgrade if AutoModelForMultimodalLM
                                      # / AutoProcessor don't resolve
        "thinking": "low",          # reasoning strength: low/medium/high/xhigh (not a bool!)
        "default_quantization": "bf16",  # native release precision; 4bit works but
                                          # untested against the perception encoder
    },
        "gemma-4-31B-it": {
        "path": os.path.join(MODELS_DIR, "gemma-4-31B-it"),
        "type": "gemma4",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,           # bool here, unlike Muse Glimmer's low/medium/high/xhigh
        "default_quantization": "bf16",  # 31B dense, same rationale as your Qwen2.5-32B entry
    },

        "deberta-v3-large": {
            "path": os.path.join(MODELS_DIR, "deberta-v3-large"), 
            "type": "encoder",
            "has_system_role": False, 
            "trust_remote_code": False, 
            "thinking": False,
            "default_quantization": "bf16"
    },
        "ModernBERT-large": {
            "path": os.path.join(MODELS_DIR, "ModernBERT-large"), 
            "type": "encoder",
            "has_system_role": False, 
            "trust_remote_code": False, 
            "thinking": False,
            "default_quantization": "bf16"
        },
    }