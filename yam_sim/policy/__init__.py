"""Policy interfaces for running trained models in the sim.

The policy module requires additional dependencies (torch).
Install with: pip install yam-sim[policy]
"""

from yam_sim.policy.base import PolicyConfig, BasePolicy

__all__ = ["PolicyConfig", "BasePolicy"]

try:
    from yam_sim.policy.lbm_policy import LBMPolicy, LBMPolicyConfig

    __all__ += ["LBMPolicy", "LBMPolicyConfig"]
except ImportError:
    pass

try:
    from yam_sim.policy.openpi_policy import OpenPIPolicy

    __all__ += ["OpenPIPolicy"]
except ImportError:
    pass
