"""
Data ingestion for the Bitext Customer Support dataset.

Strategy (most robust first):
  1. Load via the HuggingFace `datasets` library.
  2. Fall back to a direct CSV download with pandas.
  3. Fall back to a local CSV the user dropped in data/raw/.

Whatever the source, we normalise column names so the rest of the codebase can
rely on config.TEXT_COL / CATEGORY_COL / INTENT_COL / RESPONSE_COL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow `python src/data_loader.py` as well as `from src import data_loader`.
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Map many possible source column names -> our canonical names.
_COLUMN_ALIASES = {
    config.TEXT_COL: ["instruction", "utterance", "text", "message", "query", "input"],
    config.CATEGORY_COL: ["category", "label", "topic"],
    config.INTENT_COL: ["intent", "fine_label", "subcategory"],
    config.RESPONSE_COL: ["response", "answer", "reply", "output", "completion"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to our canonical names where we can find them."""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower and lower[alias] != canonical:
                rename[lower[alias]] = canonical
                break
    df = df.rename(columns=rename)
    return df


def download_dataset(force: bool = False) -> pd.DataFrame:
    """Download the Bitext dataset (or load the cached raw CSV) and return a DataFrame."""
    if config.RAW_CSV_PATH.exists() and not force:
        print(f"[data] Using cached raw CSV: {config.RAW_CSV_PATH}")
        return _normalise_columns(pd.read_csv(config.RAW_CSV_PATH))

    df = None
    # 1) HuggingFace datasets library --------------------------------------- #
    try:
        from datasets import load_dataset

        print(f"[data] Loading '{config.HF_DATASET_ID}' via HuggingFace datasets ...")
        ds = load_dataset(config.HF_DATASET_ID)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
    except Exception as exc:  # noqa: BLE001
        print(f"[data] datasets route failed ({exc.__class__.__name__}: {exc}). Trying direct CSV ...")

    # 2) Direct CSV download ------------------------------------------------- #
    if df is None:
        try:
            print(f"[data] Downloading CSV from {config.HF_CSV_FALLBACK_URL}")
            df = pd.read_csv(config.HF_CSV_FALLBACK_URL)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not download the Bitext dataset automatically.\n"
                "Fix: manually download the CSV from\n"
                f"  https://huggingface.co/datasets/{config.HF_DATASET_ID}\n"
                f"and save it to: {config.RAW_CSV_PATH}\n"
                f"Original error: {exc}"
            ) from exc

    df = _normalise_columns(df)
    _validate(df)
    config.RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CSV_PATH, index=False)
    print(f"[data] Saved raw CSV -> {config.RAW_CSV_PATH}  ({len(df):,} rows)")
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in (config.TEXT_COL, config.CATEGORY_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Edit the *_COL settings in config.py to match your file."
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: drop nulls/dupes, strip whitespace."""
    before = len(df)
    df = df.dropna(subset=[config.TEXT_COL, config.CATEGORY_COL]).copy()
    df[config.TEXT_COL] = df[config.TEXT_COL].astype(str).str.strip()
    df = df[df[config.TEXT_COL].str.len() > 0]
    df = df.drop_duplicates(subset=[config.TEXT_COL])
    df = df.reset_index(drop=True)
    print(f"[data] Cleaned: {before:,} -> {len(df):,} rows "
          f"({df[config.CATEGORY_COL].nunique()} categories)")
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    """Save parquet if possible, else CSV next to it (keeps the system dependency-light)."""
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001  (pyarrow/fastparquet missing)
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[data] parquet unavailable; wrote {csv_path} instead")


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test on the category label, saved to disk."""
    from sklearn.model_selection import train_test_split

    strat = df[config.CATEGORY_COL]
    train, temp = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=strat,
        random_state=config.RANDOM_STATE,
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[config.CATEGORY_COL],
        random_state=config.RANDOM_STATE,
    )
    for d, p in ((train, config.TRAIN_PATH), (val, config.VAL_PATH), (test, config.TEST_PATH)):
        _save(d, p)
    print(f"[data] Splits  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def _read_split(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py first.")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the saved train/val/test splits."""
    return _read_split(config.TRAIN_PATH), _read_split(config.VAL_PATH), _read_split(config.TEST_PATH)


if __name__ == "__main__":
    frame = clean(download_dataset())
    make_splits(frame)
