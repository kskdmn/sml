from __future__ import annotations

import dataclasses
import hashlib
import io
import math
from dataclasses import replace
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts import manifest as manifest_module
from sml.artifacts.manifest import (
    PayloadRef,
    canonical_json_bytes,
    file_identity,
    row_content_identity,
    structured_identity,
)

from ._manifests import (
    IDENTITY_A,
    IDENTITY_B,
    tokenizer_manifest_fixture,
)


class ExampleEnum(Enum):
    VALUE = "enum-value"


@dataclasses.dataclass(frozen=True)
class CanonicalExample:
    path: Path
    number: np.int64


def test_sml_json_v1_identity_vectors_are_stable():
    """Changing normalization of tuples, Unicode, or signed zero breaks identities."""
    left = {"z": -0.0, "a": ("雪", 1, 1.0)}
    right = {"a": ["雪", 1, 1.0], "z": 0.0}
    expected = b'{"a":["\xe9\x9b\xaa",1,1.0],"z":0}'

    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected


def test_canonical_identity_normalizes_supported_schema_values():
    """Dropping schema-value normalization would make equivalent configs diverge."""
    value = {
        "record": CanonicalExample(Path("nested/file"), np.int64(7)),
        "enum": ExampleEnum.VALUE,
        "tuple": (np.float32(1.5),),
    }

    assert canonical_json_bytes(value) == (
        b'{"enum":"enum-value","record":{"number":7,"path":"nested/file"},'
        b'"tuple":[1.5]}'
    )


@pytest.mark.parametrize(
    "value",
    [math.inf, -math.inf, math.nan, "\ud800", {"\udfff": "invalid"}],
)
def test_canonical_identity_rejects_nonfinite_numbers_and_surrogates(value):
    """Accepting unstable numeric or invalid UTF-8 values would create bad manifests."""
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        canonical_json_bytes({"value": value})


def test_file_and_structured_identity_use_exact_domain_separated_bytes():
    """Removing the domain separator would allow identities from unlike contracts to alias."""
    payload = b"exact\x00bytes"
    file_digest = hashlib.sha256(payload).hexdigest()
    structured_digest = hashlib.sha256(b'example-domain\0{"x":1}').hexdigest()

    assert file_identity(io.BytesIO(payload)) == f"sha256:{file_digest}"
    assert structured_identity("example-domain", {"x": 1}) == (
        f"sha256:{structured_digest}"
    )


def test_manifest_identity_ignores_diagnostic_locator_but_not_payload():
    """Including diagnostic paths or omitting payload IDs would destroy portability."""
    first = tokenizer_manifest_fixture(
        diagnostic_source_locator="/old/path",
        model=PayloadRef("tokenizer.model", IDENTITY_A, 10),
    )
    moved = replace(first, diagnostic_source_locator="/new/path")
    changed = replace(first, model=replace(first.model, identity=IDENTITY_B))

    assert first.recompute_identity() == moved.recompute_identity()
    assert first.recompute_identity() != changed.recompute_identity()


@pytest.mark.parametrize("dtype", ["<i4", ">i4", "int64", "uint32"])
def test_row_blocks_preserve_identity_across_chunking_and_strides(dtype):
    rows = np.arange(84, dtype=dtype).reshape(7, 12)[:, ::2]
    blocks = (rows[:2], rows[2:2], rows[2:5], rows[5:])
    assert manifest_module._row_content_identity_blocks(blocks, 7, 6) == (
        row_content_identity(rows, 7, 6)
    )


def test_row_blocks_reject_overflow_and_incorrect_count():
    with pytest.raises(ValueError, match="fit int32"):
        manifest_module._row_content_identity_blocks(
            (np.array([[0, 2**31]], dtype=np.uint32),), 1, 2
        )
    with pytest.raises(ValueError, match="row count mismatch"):
        manifest_module._row_content_identity_blocks(
            (np.zeros((3, 2), dtype=np.int32),), 4, 2
        )


def test_row_content_identity_pins_shape_count_order_and_little_endian_int32():
    """Changing row order or metadata encoding must change the semantic data digest."""
    rows = [np.array([1, 2], dtype=np.int64), np.array([-3, 4], dtype=np.int16)]
    digest = hashlib.sha256()
    digest.update(b"sml-row-content-v1\0")
    digest.update((2).to_bytes(8, "little"))
    digest.update((2).to_bytes(8, "little"))
    digest.update(np.array([1, 2], dtype="<i4").tobytes())
    digest.update(np.array([-3, 4], dtype="<i4").tobytes())

    assert row_content_identity(rows, row_count=2, row_width=2) == (
        f"sha256:{digest.hexdigest()}"
    )
    assert row_content_identity(reversed(rows), row_count=2, row_width=2) != (
        f"sha256:{digest.hexdigest()}"
    )


@pytest.mark.parametrize("dtype", ["<i4", ">i4", "i2", "u2", "u4", "i8", "u8"])
def test_row_content_identity_preserves_encoding_for_strided_integer_rows(dtype):
    values = np.array([[0, 99, 1, 99], [123, 99, 32767, 99]], dtype=dtype)
    rows = values[:, ::2]
    expected = np.array([[0, 1], [123, 32767]], dtype="<i4")

    assert not rows.flags.c_contiguous
    assert row_content_identity(rows, 2, 2) == row_content_identity(expected, 2, 2)


@pytest.mark.parametrize(
    ("dtype", "minimum"),
    [("<i4", -(2**31)), (">i4", -(2**31)), ("i8", -(2**31)), ("u4", 0), ("u8", 0)],
)
def test_row_content_identity_accepts_int32_boundaries(dtype, minimum):
    rows = np.array([[minimum, 2**31 - 1]], dtype=dtype)

    assert row_content_identity(rows, 1, 2) == row_content_identity(
        rows.astype("<i4"), 1, 2
    )


@pytest.mark.parametrize(
    ("rows", "row_count", "row_width", "message"),
    [
        ([np.array([1, 2], dtype=np.int32)], 2, 2, "row count"),
        ([np.array([1, 2], dtype=np.int32)], 1, 3, "row width|shape"),
        ([np.array([[1, 2]], dtype=np.int32)], 1, 2, "shape"),
        ([np.array([1.0, 2.0])], 1, 2, "integer"),
        ([np.array([True, False])], 1, 2, "integer"),
        ([np.array([0, 2**31], dtype=np.int64)], 1, 2, "int32"),
        ([np.array([0, -(2**31) - 1], dtype=np.int64)], 1, 2, "int32"),
        ([np.array([0, 2**31], dtype=np.uint32)], 1, 2, "int32"),
        ([np.array([0, 2**63], dtype=np.uint64)], 1, 2, "int32"),
    ],
)
def test_row_content_identity_rejects_count_width_shape_dtype_and_range_mismatches(
    rows, row_count, row_width, message
):
    """Silently coercing malformed rows would let invalid prepared data share an ID."""
    with pytest.raises((TypeError, ValueError), match=message):
        row_content_identity(rows, row_count=row_count, row_width=row_width)
