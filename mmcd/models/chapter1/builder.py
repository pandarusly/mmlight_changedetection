
from mmseg.models.builder import MODELS
DIFFERENCE = MODELS 
ATTENTION = MODELS 
FUSE = MODELS 


def build_difference(cfg):
    """Build compressor."""
    return DIFFERENCE.build(cfg)

def build_attention(cfg):
    """Build  attention."""
    return ATTENTION.build(cfg)

def build_fuse(cfg):
    """Build  fusion."""
    return FUSE.build(cfg)