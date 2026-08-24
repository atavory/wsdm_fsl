"""Export simple CASdatasets .rda data frames to CSV.

This handles the R serialization forms used by the WSDM insurance inputs:
numeric vectors, integer vectors, string vectors, factors, and data-frame
attributes.  It intentionally avoids an R runtime so the artifact can rebuild on
hosts where only Python is available.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


R_NA_INT = -(2**31)


@dataclass
class Parser:
    data: bytes
    offset: int = 0

    def read(self, size: int) -> bytes:
        chunk = self.data[self.offset : self.offset + size]
        self.offset += size
        return chunk

    def read_int(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def read_reals(self, size: int) -> list[float]:
        return list(struct.unpack(">" + "d" * size, self.read(8 * size)))

    def read_ints(self, size: int) -> list[int]:
        return list(struct.unpack(">" + "i" * size, self.read(4 * size)))

    def read_header(self) -> None:
        magic = self.read(5)
        mode = self.read(2)
        if not magic.startswith(b"RDX") or mode != b"X\n":
            raise RuntimeError("unsupported RData header")
        version = self.read_int()
        self.read_int()
        self.read_int()
        if version >= 3:
            self.read(self.read_int())

    def read_sexp(self) -> Any:
        flags = self.read_int()
        sexp_type = flags & 255
        has_attr = bool((flags >> 9) & 1)
        has_tag = bool((flags >> 10) & 1)
        if sexp_type in (0, 254):
            return None
        if sexp_type in (2, 6):
            tag = self.read_sexp() if has_tag else None
            value = self.read_sexp()
            next_node = self.read_sexp()
            attr = self.read_sexp() if has_attr else None
            return ("LIST", tag, value, next_node, attr)
        if sexp_type == 1:
            return ("SYM", self.read_sexp())
        if sexp_type == 9:
            size = self.read_int()
            return None if size < 0 else self.read(size).decode("utf-8", "replace")
        if sexp_type in (10, 13):
            size = self.read_int()
            values = self.read_ints(size)
            attr = self.read_sexp() if has_attr else None
            return ("INT", values, attr)
        if sexp_type == 14:
            size = self.read_int()
            values = self.read_reals(size)
            attr = self.read_sexp() if has_attr else None
            return ("REAL", values, attr)
        if sexp_type == 16:
            size = self.read_int()
            values = [self.read_sexp() for _ in range(size)]
            attr = self.read_sexp() if has_attr else None
            return ("STR", values, attr)
        if sexp_type == 19:
            size = self.read_int()
            values = [self.read_sexp() for _ in range(size)]
            attr = self.read_sexp() if has_attr else None
            return ("VEC", values, attr)
        if sexp_type == 255:
            ref_index = flags >> 8
            if ref_index == 0:
                ref_index = self.read_int()
            return ("REF", ref_index)
        raise RuntimeError(f"unsupported R SEXP type {sexp_type} at {self.offset - 4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decompress(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw.startswith(b"BZh"):
        return bz2.decompress(raw)
    if raw.startswith(b"\x1f\x8b"):
        return gzip.decompress(raw)
    return raw


def symbol_name(value: Any) -> str | None:
    if isinstance(value, tuple) and value[0] == "SYM":
        return value[1]
    if isinstance(value, str):
        return value
    return None


def list_attributes(node: Any) -> list[tuple[str | None, Any]]:
    result: list[tuple[str | None, Any]] = []
    while isinstance(node, tuple) and node[0] == "LIST":
        result.append((symbol_name(node[1]), node[2]))
        node = node[3]
    return result


def attr_value(node: Any, name: str) -> Any:
    for key, value in list_attributes(node):
        if key == name:
            return value
    return None


def string_values(node: Any) -> list[str]:
    if isinstance(node, tuple) and node[0] == "STR":
        return ["" if value is None else str(value) for value in node[1]]
    return []


def factor_levels(node: Any) -> list[str] | None:
    for _, value in list_attributes(node):
        values = string_values(value)
        if values and values != ["factor"]:
            return values
    return None


def scalar_text(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return "" if value == R_NA_INT else str(value)
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def column_values(column: Any) -> list[str]:
    kind = column[0]
    values = column[1]
    attrs = column[2] if len(column) > 2 else None
    levels = factor_levels(attrs)
    if kind == "INT" and levels is not None:
        return [
            ""
            if value == R_NA_INT or value <= 0 or value > len(levels)
            else levels[value - 1]
            for value in values
        ]
    return [scalar_text(value) for value in values]


def load_data_frame(path: Path) -> tuple[list[str], list[list[str]]]:
    parser = Parser(decompress(path))
    parser.read_header()
    root = parser.read_sexp()
    if parser.offset != len(parser.data):
        raise RuntimeError("RData parser stopped before end of file")
    if not (isinstance(root, tuple) and root[0] == "LIST"):
        raise RuntimeError("expected saved-object pairlist")
    frame = root[2]
    if not (isinstance(frame, tuple) and frame[0] == "VEC"):
        raise RuntimeError("expected data-frame vector")
    names_node = attr_value(frame[2], "names")
    names = string_values(names_node)
    columns = frame[1]
    if len(names) != len(columns):
        raise RuntimeError("data-frame names do not match columns")
    rendered = [column_values(column) for column in columns]
    row_count = len(rendered[0]) if rendered else 0
    if any(len(column) != row_count for column in rendered):
        raise RuntimeError("data-frame columns have different lengths")
    return names, rendered


def main() -> None:
    args = parse_args()
    names, columns = load_data_frame(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for row_index in range(len(columns[0])):
            writer.writerow([column[row_index] for column in columns])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
