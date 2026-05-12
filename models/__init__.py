"""Model registry."""

from .transformer import Transformer, TransformerConfig

__all__ = ["Transformer", "TransformerConfig"]

try:
    from .delta_net import DeltaNet, DeltaNetWrapperConfig

    __all__ += ["DeltaNet", "DeltaNetWrapperConfig"]
except ImportError:
    pass

try:
    from .gla import GLA, GLAWrapperConfig

    __all__ += ["GLA", "GLAWrapperConfig"]
except ImportError:
    pass
