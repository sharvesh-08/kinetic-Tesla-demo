"""Embedded Redis server using fakeredis for environments without standalone redis-server."""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse
from fakeredis import TcpFakeServer


def get_port_from_env() -> int:
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        parsed = urlparse(redis_url)
        if parsed.port:
            return parsed.port
    return 6380


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else get_port_from_env()
    server = TcpFakeServer(("127.0.0.1", port))
    print(f"[OK] Embedded fakeredis server listening on 127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
