"""CLI entry point: python -m smriti_memcore.ui"""
import argparse
from smriti_memcore.ui.server import launch

def main():
    parser = argparse.ArgumentParser(
        prog="smriti_ui",
        description="🏛️  Smriti Memory Browser — visualize your agent's Semantic Palace",
    )
    parser.add_argument(
        "--storage", "-s",
        default="~/.smriti/global",
        help="Path to Smriti storage directory (default: ~/.smriti/global)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7799,
        help="Port to serve the UI on (default: 7799)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically",
    )
    args = parser.parse_args()
    launch(storage_path=args.storage, port=args.port, open_browser=not args.no_browser)

if __name__ == "__main__":
    main()
