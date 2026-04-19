#!/usr/bin/env python3
"""Launcher for lm-eval with vLLM 0.19+.

Patches:
1. __main__ guard required by vLLM's spawn multiprocessing
2. EngineArgs patched to ignore swap_space (removed in vLLM 0.19)
"""
if __name__ == "__main__":
    from vllm.engine.arg_utils import EngineArgs
    _orig_init = EngineArgs.__init__

    def _patched_init(self, **kwargs):
        kwargs.pop("swap_space", None)
        _orig_init(self, **kwargs)

    EngineArgs.__init__ = _patched_init

    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
