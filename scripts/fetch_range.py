"""Finish a partially downloaded file using an HTTP Range request."""

from __future__ import annotations

import argparse
import os
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    args = parser.parse_args()

    offset = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    request = urllib.request.Request(
        args.url,
        headers={"Range": f"bytes={offset}-", "User-Agent": "pip-compatible-downloader"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if offset and status != 206:
            raise RuntimeError(f"Server ignored Range request at byte {offset}: HTTP {status}")
        with open(args.output, "ab" if offset else "wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    print(f"downloaded {os.path.getsize(args.output)} bytes to {args.output}")


if __name__ == "__main__":
    main()
