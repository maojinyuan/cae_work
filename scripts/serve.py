"""Internal entry point for the CLI's detached local service."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.local_service import serve

if __name__ == '__main__':
    serve()
