#!/usr/bin/env python
"""Download the Bitext dataset, clean it, and write stratified train/val/test splits."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_loader  # noqa: E402


def main() -> None:
    df = data_loader.download_dataset()
    df = data_loader.clean(df)
    data_loader.make_splits(df)
    print("\nDone. Splits written to data/processed/.")


if __name__ == "__main__":
    main()
