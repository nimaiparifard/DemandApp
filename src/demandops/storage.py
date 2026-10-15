"""Tiny object-store abstraction: a local folder or any S3-API store (SeaweedFS, MinIO, Arvan, AWS).

URIs look like `s3://bucket/prefix` or a plain local path. Swapping the store (e.g. for AWS S3)
changes only the environment (endpoint URL / credentials), not the code.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


def _filesystem(uri: str, endpoint_url: str | None) -> tuple[pafs.FileSystem, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        endpoint = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint_override"] = endpoint.replace("http://", "").replace("https://", "")
            kwargs["scheme"] = "https" if endpoint.startswith("https") else "http"
        key, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
        if key and secret:
            kwargs.update(access_key=key, secret_key=secret)
        return pafs.S3FileSystem(**kwargs), f"{parsed.netloc}{parsed.path}"
    return pafs.LocalFileSystem(), os.path.abspath(uri)


class ObjectStore:
    def __init__(self, uri: str, endpoint_url: str | None = None) -> None:
        self.fs, self.root = _filesystem(uri, endpoint_url)
        if isinstance(self.fs, pafs.LocalFileSystem):
            os.makedirs(self.root, exist_ok=True)

    def _path(self, name: str) -> str:
        return f"{self.root.rstrip('/')}/{name}"

    def write_parquet(self, df: pd.DataFrame, name: str) -> None:
        pq.write_table(_to_table(df), self._path(name), filesystem=self.fs)

    def read_parquet(self, name: str) -> pd.DataFrame:
        return pq.read_table(self._path(name), filesystem=self.fs).to_pandas()

    def write_json(self, obj: Any, name: str) -> None:
        with self.fs.open_output_stream(self._path(name)) as f:
            f.write(json.dumps(obj).encode())

    def read_json(self, name: str) -> Any:
        with self.fs.open_input_stream(self._path(name)) as f:
            return json.loads(io.BytesIO(f.read()).read())

    def exists(self, name: str) -> bool:
        return self.fs.get_file_info(self._path(name)).type != pafs.FileType.NotFound

    def list(self, prefix: str = "") -> list[str]:
        sel = pafs.FileSelector(
            self._path(prefix).rstrip("/"), allow_not_found=True, recursive=True
        )
        base = self.root.rstrip("/") + "/"
        return sorted(
            i.path.replace("\\", "/").removeprefix(base.replace("\\", "/"))
            for i in self.fs.get_file_info(sel)
            if i.is_file
        )


def _to_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)
