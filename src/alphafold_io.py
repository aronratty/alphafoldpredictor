from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

AFDB_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def get_prediction_metadata(
    accession: str,
    *,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch AlphaFold DB prediction metadata for a UniProt accession.

    AlphaFold DB currently returns one or more prediction records.  We keep
    this function tolerant of either a list or a single JSON object so the
    rest of the code is insulated from minor response-shape changes.
    """
    accession = accession.strip().upper()
    client = session or requests
    url = AFDB_PREDICTION_API.format(accession=accession)

    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unexpected AlphaFold API response type: {type(payload)!r}")

    if not records:
        raise LookupError(f"No AlphaFold DB prediction found for {accession}")

    return records


def choose_prediction_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Choose a prediction record.

    Most UniProt accessions have one AlphaFold DB entry. If multiple records
    are returned, prefer the record that starts earliest in the sequence.
    The code supports both the current `sequenceStart` field and the former
    `uniprotStart` name for backward compatibility with cached metadata.
    """
    if not records:
        raise ValueError("records must not be empty")

    def start_position(record: dict[str, Any]) -> int:
        value = record.get("sequenceStart", record.get("uniprotStart", 1))
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    return min(records, key=start_position)


def _find_structure_url(record: dict[str, Any], fmt: str) -> tuple[str, str]:
    fmt = fmt.lower()

    candidates: dict[str, list[tuple[str, str]]] = {
        "cif": [
            ("cifUrl", ".cif"),
            ("mmcifUrl", ".cif"),
        ],
        "pdb": [
            ("pdbUrl", ".pdb"),
        ],
    }

    if fmt not in candidates:
        raise ValueError("fmt must be 'cif' or 'pdb'")

    for key, suffix in candidates[fmt]:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value, suffix

    # If the preferred format is absent, fall back to the other text format.
    fallback = "pdb" if fmt == "cif" else "cif"
    for key, suffix in candidates[fallback]:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value, suffix

    known_keys = ", ".join(sorted(record.keys()))
    raise KeyError(
        "AlphaFold API record did not contain a supported structure URL. "
        f"Available fields: {known_keys}"
    )


def download_structure(
    accession: str,
    output_dir: str | Path,
    *,
    fmt: str = "cif",
    overwrite: bool = False,
    timeout: float = 60.0,
) -> tuple[Path, dict[str, Any]]:
    """
    Fetch metadata through the AlphaFold REST API and download the structure.

    Returns
    -------
    path:
        Local structure file.
    record:
        The selected AlphaFold prediction metadata record.
    """
    accession = accession.strip().upper()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = get_prediction_metadata(accession, timeout=timeout)
    record = choose_prediction_record(records)
    structure_url, suffix = _find_structure_url(record, fmt)

    path = output_dir / f"{accession}{suffix}"
    if path.exists() and not overwrite:
        return path, record

    response = requests.get(structure_url, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)

    return path, record
