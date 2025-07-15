#!/usr/bin/env python3
"""
Simple runner for AI vs AI Chess Game
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_vs_ai_pygame import main

if __name__ == "__main__":
    main()