"""Evaluation harness for Quark models."""

import os

import hydra
from checkpoint_utils import match_state_dict_keys, maybe_load_checkpoint
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from models import DeltaNet, DeltaNetWrapperConfig, Transformer, TransformerConfig
from omegaconf import OmegaConf


def build_model_for_eval(config_name, model_overrides=None):
    """Build model using Hydra config, similar to train.py.

    Args:
        config_name (str): Name of the model config to use.
        model_overrides (dict, optional): Model overrides.

    Returns:
        tuple: (model, cfg)
    """
    with hydra.initialize(version_base=None, config_path="../configs"):
        # We override the model if specified, e.g., model=delta_net
        overrides = [f"model={config_name}"]
        if model_overrides:
            for k, v in model_overrides.items():
                overrides.append(f"model.{k}={v}")

        cfg = hydra.compose(config_name="config", overrides=overrides)

        vocab_size = cfg.data.vocab_size
        model_type = cfg.model.model_type

        if model_type == "delta_net":
            config = DeltaNetWrapperConfig(
                vocab_size=vocab_size,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_layers,
                num_heads=cfg.model.num_heads,
                expand_k=cfg.model.expand_k,
                expand_v=cfg.model.expand_v,
                use_gate=cfg.model.use_gate,
                use_beta=cfg.model.use_beta,
                use_short_conv=cfg.model.use_short_conv,
                conv_size=cfg.model.conv_size,
            )
            return DeltaNet(config), cfg

        config = TransformerConfig(
            vocab_size=vocab_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            block_size=cfg.data.seq_len,
            dropout=cfg.model.dropout,
            bias=cfg.model.bias,
        )
        return Transformer(config), cfg


@register_model("quark")
class QuarkLM(HFLM):
    """HuggingFace LM wrapper for Quark models."""

    def __init__(
        self,
        checkpoint,
        config="delta_net",
        tokenizer="EleutherAI/gpt-neox-20b",
        device="cuda",
        batch_size=1,
        dtype="bfloat16",
        **kwargs,
    ):
        """Initialize QuarkLM for evaluation."""
        # 1. Build model and load checkpoint
        # Extract Quark-specific kwargs to avoid passing them to HFLM
        custom_kwargs = [
            "hidden_size",
            "num_layers",
            "num_heads",
            "expand_k",
            "expand_v",
            "use_beta",
            "use_gate",
            "use_short_conv",
            "conv_size",
            "step",
        ]

        model_overrides = {}
        for k in custom_kwargs:
            if k in kwargs:
                model_overrides[k] = kwargs.pop(k)

        model_wrapper, cfg = build_model_for_eval(config, model_overrides)

        # Configure cfg for maybe_load_checkpoint
        cfg_override = OmegaConf.create({"checkpoint": {"resume": True}})
        cfg = OmegaConf.merge(cfg, cfg_override)

        # Determine out_dir and exp_name
        if os.path.isabs(checkpoint):
            cfg_override = OmegaConf.create(
                {
                    "out_dir": os.path.dirname(checkpoint),
                    "checkpoint": {"resume_exp_name": os.path.basename(checkpoint)},
                }
            )
            cfg = OmegaConf.merge(cfg, cfg_override)
        else:
            cfg_override = OmegaConf.create(
                {
                    "checkpoint": {"resume_exp_name": checkpoint},
                }
            )
            cfg = OmegaConf.merge(cfg, cfg_override)

        # Parse step if provided
        step = model_overrides.get("step")
        if step is not None:
            cfg_override = OmegaConf.create(
                {
                    "checkpoint": {"resume_step": int(step)},
                }
            )
            cfg = OmegaConf.merge(cfg, cfg_override)

        ckpt = maybe_load_checkpoint(cfg)
        if ckpt is None:
            raise ValueError(f"Failed to load checkpoint for {checkpoint}")

        # Apply state dict
        state_dict = ckpt["state_dict"]
        state_dict = match_state_dict_keys(state_dict, model_wrapper.state_dict())
        model_wrapper.load_state_dict(state_dict)

        if hasattr(model_wrapper, "model"):
            model = model_wrapper.model
        else:
            model = model_wrapper

        model.to(device=device, dtype=dtype)
        model.eval()

        # 2. Initialize HFLM
        # We pass the pre-instantiated model and tokenizer
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def max_length(self):
        """Get the max length of the model."""
        # Try to get from kwargs or default to 2048
        if hasattr(self, "_max_length") and getattr(self, "_max_length") is not None:
            return self._max_length
        return 2048
