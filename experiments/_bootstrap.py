"""Add the project root to sys.path so experiments can import ``src``."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
