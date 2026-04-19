#!/usr/bin/env python3
"""Patches transformers 5.x / vLLM 0.10.x dataclass incompatibility, then runs lm-eval.

vLLM 0.10.1 defines PretrainedConfig subclasses (e.g. DeepseekVLV2Config) with
field ordering that violates Python dataclass rules. transformers 5.x auto-applies
@dataclass to all PretrainedConfig subclasses via __init_subclass__, causing TypeError.
This patch catches and skips the failing decoration.
"""
import transformers.configuration_utils as _cfg

_orig = _cfg.PretrainedConfig.__init_subclass__

@classmethod
def _patched(cls, **kwargs):
    try:
        _orig.__func__(cls, **kwargs)
    except TypeError:
        pass

_cfg.PretrainedConfig.__init_subclass__ = _patched

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
