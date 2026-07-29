from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "provenance" / "source_integrity_sha256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        source = row["source_id"]
        path = PROJECT_ROOT / row["local_file"]
        if not path.is_file():
            failures.append(f"{source}: missing {path}")
            continue
        observed_size = path.stat().st_size
        expected_size = int(row["byte_size"])
        if observed_size != expected_size:
            failures.append(
                f"{source}: byte size {observed_size} != {expected_size}"
            )
        observed_hash = sha256(path)
        if observed_hash != row["sha256"]:
            failures.append(
                f"{source}: sha256 {observed_hash} != {row['sha256']}"
            )

    if failures:
        raise SystemExit("Source-integrity verification failed:\n- " + "\n- ".join(failures))

    print(f"Source-integrity verification passed for {len(rows)} local records.")


if __name__ == "__main__":
    main()
