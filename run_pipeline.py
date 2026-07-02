"""
run_pipeline.py
Bank Retention Intelligence Platform — Master Pipeline Runner (legacy entrypoint).

This module used to contain its own copy of the 8-step pipeline. After the
src/ refactor, the canonical orchestrator lives in src.pipeline.run_full_pipeline.
This file is kept only so legacy scripts and notebooks that still call
`python run_pipeline.py` continue to work — it is a 5-line re-export of
main.main(), not a second implementation.

Run: python run_pipeline.py
"""
from main import main

if __name__ == "__main__":
    main()
