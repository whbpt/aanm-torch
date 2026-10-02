"""AANM (Adaptive Anisotropic Network Model) morph tools, PyTorch rewrite.

Replaces the ProDy-based scripts in designer/machine/scripts/ with a pure
torch implementation that runs on CUDA (or MPS / CPU). The AANM stepping
algorithm is a faithful port of ProDy's calcAdaptiveANM(AANM_ONEWAY).
"""
import torch

__version__ = "1.1.0"

AANM_ONEWAY = 1


def pick_device(name="auto", dtype_name="float64"):
    """Resolve a torch device: 'auto' picks cuda > mps > cpu.

    MPS has no float64 support, so a float64 run on a Mac falls back to CPU
    (CUDA keeps float64). Pass dtype_name explicitly to control this.
    """
    want_mps = (name == "auto" or name == "mps") and \
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if want_mps and dtype_name != "float32":
        if name == "mps":
            raise ValueError("MPS does not support float64; use --dtype float32 or --device cpu")
        return torch.device("cpu")
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if want_mps:
        return torch.device("mps")
    return torch.device("cpu")
