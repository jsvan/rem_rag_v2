#!/usr/bin/env python3
"""
Script to run the 1922 Foreign Affairs experiment.

This will process all articles from 1922 (the founding year of Foreign Affairs)
through READING and REM cycles to test the REM RAG system.

Usage:
    python scripts/run_1922_experiment.py
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.experiments.year_1922 import main

if __name__ == "__main__":
    print("Starting 1922 Foreign Affairs Experiment...")
    print("This will process articles from Foreign Affairs' founding year.")
    print("Make sure you have set OPENAI_API_KEY in your environment.\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        raise