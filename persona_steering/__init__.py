"""Persona-Conditional Steering Vectors.

Investigating whether steering vectors for the same trait change
depending on which persona the model is operating under.
"""

from pathlib import Path

from dotenv import load_dotenv

# Auto-load .env from project root (parent of this package)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

__version__ = "0.1.0"
