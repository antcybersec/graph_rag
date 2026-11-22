"""Normalising TigerGraph query results into plain rows.

An installed query returns one object per PRINT, and a projected PRINT prefixes
every attribute with the vertex set's name:

    PRINT Res[Res.games_year, Res.nations];
    -> {"Res": [{"v_id": "...", "attributes": {"Res.games_year": 1992, ...}}]}

Stripping that prefix in one place keeps it out of every caller. Doing it at
each call site is how you end up with a KeyError in the middle of a benchmark
run, which is exactly what happened the first time.
"""
from __future__ import annotations

from typing import Any


def rows(result: Any, block: str | None = None) -> list[dict]:
    """Every vertex/edge row in a result, as flat dicts.

    `block` selects one PRINT by name; omitted, all list-valued blocks are
    concatenated. `v_id` and `v_type` are preserved alongside the attributes,
    since a citation needs the id and the type disambiguates mixed results.
    """
    out: list[dict] = []
    for chunk in (result or []):
        if not isinstance(chunk, dict):
            continue
        for name, value in chunk.items():
            if block is not None and name != block:
                continue
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                attrs = item.get("attributes", item)
                flat = {}
                for key, val in (attrs or {}).items():
                    # "Res.games_year" -> "games_year"; a plain attribute name
                    # (unprojected PRINT) passes through untouched.
                    flat[key.split(".", 1)[1] if "." in key else key] = val
                for meta in ("v_id", "v_type", "e_type", "from_id", "to_id"):
                    if meta in item:
                        flat.setdefault(meta, item[meta])
                out.append(flat)
    return out


def scalar(result: Any, name: str, default=None):
    """A PRINT of a scalar or map, by name."""
    for chunk in (result or []):
        if isinstance(chunk, dict) and name in chunk:
            return chunk[name]
    return default
