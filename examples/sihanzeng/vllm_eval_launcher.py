#!/usr/bin/env python3
"""Launcher for lm-eval with vLLM 0.19+ (requires __main__ guard for spawn multiprocessing)."""
if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
