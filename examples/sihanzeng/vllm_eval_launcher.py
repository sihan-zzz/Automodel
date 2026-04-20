#!/usr/bin/env python3
"""Launcher for lm-eval with vLLM 0.19+ and Gemma4 thinking mode.

Patches:
1. __main__ guard required by vLLM's spawn multiprocessing
2. EngineArgs patched to ignore swap_space (removed in vLLM 0.19)
3. VLLM model patched to pass enable_thinking=True to chat template
"""
if __name__ == "__main__":
    from vllm.engine.arg_utils import EngineArgs
    _orig_ea = EngineArgs.__init__

    def _patched_ea(self, **kwargs):
        kwargs.pop("swap_space", None)
        _orig_ea(self, **kwargs)

    EngineArgs.__init__ = _patched_ea

    import lm_eval.models.vllm_causallms as _mod
    _orig_init = _mod.VLLM.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.chat_template_kwargs = {"enable_thinking": True}

    _mod.VLLM.__init__ = _patched_init

    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
