#!/usr/bin/env python3
"""Launcher for lm-eval with vLLM 0.19+.

Patches:
1. __main__ guard required by vLLM's spawn multiprocessing
2. swap_space removed from EngineArgs (API changed in vLLM 0.19)
"""
if __name__ == "__main__":
    import lm_eval.models.vllm_causallms as _mod
    _OrigVLLM = _mod.VLLM

    class PatchedVLLM(_OrigVLLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_args.pop("swap_space", None)

    _mod.VLLM = PatchedVLLM

    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
