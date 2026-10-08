"""Registry of the accepted style encoders.

Each accepted style encoder has one identity that both a machine and a
reader can use: the Hub id and revision, the sha256 of every file of the
Hub snapshot, and a description. A key never changes once assigned, and
an encoder with new bytes gets a new key. The reference copy of
``file_sha256`` is the Hub snapshot.
"""

import json
import os

_REGISTRY_FILENAME = "style_encoders.json"

_registry_cache = None


def _registry_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _REGISTRY_FILENAME)


def _load_registry():
    global _registry_cache
    if _registry_cache is None:
        with open(_registry_path(), encoding="utf-8") as handle:
            _registry_cache = json.load(handle)
    return _registry_cache


def check_encoder_entry(key, entry):
    """Check one registry entry against the schema and return it.

    An entry needs ``revision``, ``file_sha256`` (a non-empty dict) and
    ``description``. It needs ``hub_id`` unless ``local_only`` names the
    machine that holds the bytes. Raises ValueError that names the key
    and the missing field otherwise.
    """
    for field in ("revision", "file_sha256", "description"):
        if field not in entry:
            raise ValueError(f"style encoder {key!r} has no {field!r}")
    if not isinstance(entry["file_sha256"], dict) or not entry["file_sha256"]:
        raise ValueError(f"style encoder {key!r} has an empty file_sha256 map")
    if "hub_id" not in entry and "local_only" not in entry:
        raise ValueError(
            f"style encoder {key!r} needs hub_id unless local_only is set"
        )
    return entry


def get_encoder(key):
    """Return the registry entry for `key`, checked against the schema.

    Raises KeyError with the list of known keys on an unknown key.
    """
    registry = _load_registry()
    known_keys = sorted(k for k in registry if k != "exclude")
    if key not in registry or key == "exclude":
        raise KeyError(f"unknown style encoder {key!r}; known keys: {known_keys}")
    return check_encoder_entry(key, registry[key])


def get_excluded_filenames():
    """Return the snapshot filenames that are never hashed."""
    return list(_load_registry().get("exclude", []))
