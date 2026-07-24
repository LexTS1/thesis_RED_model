"""Recover canonical CORDEX and PVGIS inputs from legacy model_v3."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .load_cordex import load_config, resolve_config_path, sha256_file


LOGGER = logging.getLogger("climate.recover_model_v3")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _copy_verified(
    source: Path, destination: Path, expected_hash: str, overwrite: bool
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Legacy source does not exist: {source}")
    source_hash = sha256_file(source)
    if source_hash != expected_hash:
        raise ValueError(
            f"Legacy source hash mismatch for {source}: expected {expected_hash}, got {source_hash}"
        )

    if destination.exists():
        destination_hash = sha256_file(destination)
        if destination_hash == expected_hash:
            LOGGER.info("Verified existing canonical copy: %s", destination)
            return
        if not overwrite:
            raise FileExistsError(
                f"Destination exists with different content: {destination}. "
                "Pass --overwrite only after confirming the configured source."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Copied file failed hash verification: {temporary}")
    temporary.replace(destination)
    LOGGER.info("Recovered %s", destination)


def recover_inputs(
    config: Mapping[str, Any], source_root: Path, overwrite: bool = False
) -> None:
    """Copy and verify every configured CORDEX and PVGIS input."""

    for key, spec in config["sources"].items():
        LOGGER.info("Recovering configured CORDEX source %s", key)
        _copy_verified(
            source_root / spec["legacy_csv"],
            resolve_config_path(config, spec["csv"]),
            str(spec["csv_sha256"]),
            overwrite,
        )
        _copy_verified(
            source_root / spec["legacy_metadata"],
            resolve_config_path(config, spec["metadata"]),
            str(spec["metadata_sha256"]),
            overwrite,
        )

    observed = config.get("observed_weather")
    if observed:
        observed_specs = {"horizontal": observed["horizontal"], **observed["facades"]}
        for key, spec in observed_specs.items():
            LOGGER.info("Recovering configured PVGIS source %s", key)
            _copy_verified(
                source_root / spec["legacy_csv"],
                resolve_config_path(config, spec["csv"]),
                str(spec["csv_sha256"]),
                overwrite,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover byte-identical CORDEX and PVGIS inputs from legacy model_v3."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Override the legacy model_v3 root recorded in config.yaml.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )
    try:
        config = load_config(args.config)
        configured_root = Path(config["provenance"]["legacy_model_root"])
        source_root = (args.source_root or configured_root).expanduser().resolve()
        recover_inputs(config, source_root, overwrite=args.overwrite)
    except Exception as exc:
        LOGGER.error("Climate-input recovery failed: %s", exc)
        return 1
    LOGGER.info("All canonical CORDEX and PVGIS inputs are present and hash-verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
