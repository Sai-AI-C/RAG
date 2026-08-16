"""
OmniDoc-RAG Application Entrypoint
Allows starting the Streamlit application or executing the ingestion pipeline from CLI.
"""

import sys
import os
import argparse
from src.utils.helpers import load_app_config


def run_streamlit():
    """Launch the Streamlit web interface."""
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    print("🚀 Starting OmniDoc AI Streamlit Application...")
    subprocess.run(cmd)


def run_ingestion(data_dir: str = "./PDF_Data"):
    """Run incremental document ingestion into ChromaDB."""
    from src.ingestion.loader import process_directory_incrementally
    print(f"🔄 Starting Document Ingestion from '{data_dir}'...")
    process_directory_incrementally(root_path=data_dir)


def main():
    parser = argparse.ArgumentParser(description="OmniDoc-RAG: Engineering Academic Assistant")
    parser.add_argument(
        "--mode",
        choices=["app", "ingest", "test"],
        default="app",
        help="Execution mode: 'app' (default: Streamlit UI), 'ingest' (process PDFs), 'test' (run test suite)"
    )
    parser.add_argument("--data-dir", default="./PDF_Data", help="Path to PDF directory for ingestion")

    args = parser.parse_args()

    if args.mode == "app":
        run_streamlit()
    elif args.mode == "ingest":
        run_ingestion(args.data_dir)
    elif args.mode == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)


if __name__ == "__main__":
    main()
