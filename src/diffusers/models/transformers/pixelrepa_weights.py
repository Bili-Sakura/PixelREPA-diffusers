"""State-dict remapping from legacy PixelREPA checkpoints to native PixelREPATransformer2DModel keys."""

from __future__ import annotations

from typing import Dict

import torch

PIXELREPA_PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "JiT-B/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 4,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-L/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 8,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-H/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1280,
        "num_layers": 32,
        "num_attention_heads": 16,
        "bottleneck_dim": 256,
        "in_context_len": 32,
        "in_context_start": 10,
        "attention_dropout": 0.0,
        "dropout": 0.2,
    },
}


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map wrapper/backbone keys from legacy PixelREPA checkpoints to native PixelREPATransformer2DModel keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("transformer.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.")
        if new_key.startswith("final_layer."):
            new_key = new_key.replace("final_layer.norm_final", "norm_final")
            new_key = new_key.replace("final_layer.linear", "linear_final")
            new_key = new_key.replace("final_layer.adaLN_modulation", "adaLN_modulation_final")

        remapped[new_key] = value
    return remapped


def config_from_legacy(config: Dict[str, object]) -> Dict[str, object]:
    """Build native config kwargs from a legacy config.json dict."""
    model_type = config.get("model_type") or config.get("model_name") or config.get("model")
    if model_type not in PIXELREPA_PRESET_CONFIGS:
        raise ValueError(f"Unknown PixelREPA preset '{model_type}'. Known: {list(PIXELREPA_PRESET_CONFIGS)}")

    preset = dict(PIXELREPA_PRESET_CONFIGS[model_type])
    preset["num_classes"] = int(config.get("num_class_embeds") or config.get("num_classes") or config.get("class_num") or 1000)

    if config.get("attention_dropout") is not None:
        preset["attention_dropout"] = float(config["attention_dropout"])
    if config.get("dropout") is not None:
        preset["dropout"] = float(config["dropout"])
    if config.get("sample_size") is not None:
        preset["sample_size"] = int(config["sample_size"])

    return preset
