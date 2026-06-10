"""Export CUAD PDFs to `data/contracts/`.

The HuggingFace `theatticusproject/cuad` dataset currently exposes the `pdf`
column as `pdfplumber.PDF` objects (not raw bytes). This script resolves the
underlying local cache path and copies each physical PDF to the project folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    out = Path("data/contracts")
    out.mkdir(parents=True, exist_ok=True)

    print("Loading CUAD dataset from HuggingFace cache...")
    ds = load_dataset(
        "theatticusproject/cuad",
        split="train",
        verification_mode="no_checks",
    )

    exported = 0
    skipped = 0
    for index, row in enumerate(ds):
        pdf = row.get("pdf")
        source_path = getattr(getattr(pdf, "stream", None), "name", None)
        if source_path and Path(source_path).is_file():
            source_name = Path(source_path).name
            dest = out / f"cuad_{index:04d}_{source_name}"
            shutil.copy2(source_path, dest)
            exported += 1
        else:
            skipped += 1

    print(f"Done. exported={exported}, skipped={skipped}")
    print(f"PDFs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
