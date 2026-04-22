#!/usr/bin/env python3
"""Minimal launcher — only patches swap_space for vLLM 0.19 compat.
enable_thinking is handled natively by lm-eval via model_args."""
if __name__ == "__main__":
    from vllm.engine.arg_utils import EngineArgs
    _orig = EngineArgs.__init__
    def _patched(self, **kw):
        kw.pop("swap_space", None)
        _orig(self, **kw)
    EngineArgs.__init__ = _patched

    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
