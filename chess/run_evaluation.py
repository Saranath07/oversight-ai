#!/usr/bin/env python3
"""
Simple runner for Chess Model Evaluation
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_evaluation import main

if __name__ == "__main__":
    main()