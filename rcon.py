"""Minimal, dependency-free RCON client for Minecraft servers.

Usage as a library
------------------
    from rcon import RconClient

    with RconClient("localhost", 25575, "password") as rcon:
        print(rcon.command("list"))
        print(rcon.command("gamerule doFireTick false"))

Usage from the command line (quick one-offs)
--------------------------------------------
    python rcon.py "list"
    python rcon.py --host localhost --port 25575 --password voxeltree "dumpnoise 4"
    python rcon.py --password voxeltree "gamerule doFireTick false"
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys

__all__ = ["RconClient", "RconError"]

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_RCON_LOGIN = 3
_RCON_COMMAND = 2
_MAX_PAYLOAD = 4096


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RconError(Exception):
    """Raised for connection, authentication, or protocol errors."""


class RconClient:
    """Synchronous, single-threaded RCON client over TCP.

    Implements the Minecraft RCON protocol (Source RCON).  Use as a context
    manager for automatic connect/close::

        with RconClient("localhost", 25575, "password") as rcon:
            response = rcon.command("list")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 25575,
        password: str = "",
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self._password = password
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._req_id = 1

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the TCP connection and authenticate."""
        try:
            self._sock = socket.create_connection((self.host, self.port), self._timeout)
        except OSError as exc:
            raise RconError(
                f"Cannot connect to {self.host}:{self.port} — {exc}\n"
                "Make sure the server is running with enable-rcon=true in server.properties."
            ) from exc

        self._send_packet(_RCON_LOGIN, self._password)
        resp_id, _, _ = self._recv_packet()
        if resp_id == -1:
            raise RconError("RCON authentication failed — wrong password?")

    def command(self, cmd: str) -> str:
        """Send *cmd* and return the server's response string.

        The leading slash is optional; both ``"list"`` and ``"/list"`` work.
        """
        if self._sock is None:
            raise RconError("Not connected — call connect() or use as a context manager.")
        self._send_packet(_RCON_COMMAND, cmd)
        _, _, payload = self._recv_packet()
        return payload

    def close(self) -> None:
        """Close the underlying socket."""
        if self._sock:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "RconClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_packet(self, ptype: int, payload: str) -> None:
        assert self._sock is not None
        data = payload.encode("utf-8") + b"\x00\x00"
        header = struct.pack("<iii", len(data) + 8, self._req_id, ptype)
        self._sock.sendall(header + data)
        self._req_id += 1

    def _recv_packet(self) -> tuple[int, int, str]:
        raw_len = self._recv_exact(4)
        (pkt_len,) = struct.unpack("<i", raw_len)
        body = self._recv_exact(pkt_len)
        req_id, ptype = struct.unpack("<ii", body[:8])
        payload = body[8:-2].decode("utf-8", errors="replace")
        return req_id, ptype, payload

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise RconError("Connection closed by server.")
            buf += chunk
        return buf


# ---------------------------------------------------------------------------
# CLI entry point — useful for quick one-offs and pipeline debugging
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Send a single RCON command and print the response.",
        epilog="Example: python rcon.py --password voxeltree 'gamerule doFireTick false'",
    )
    p.add_argument("command", help="Command to send (leading slash optional)")
    p.add_argument("--host", default="localhost", metavar="HOST")
    p.add_argument("--port", type=int, default=25575, metavar="PORT")
    p.add_argument("--password", default="", metavar="PASS")
    p.add_argument("--timeout", type=float, default=10.0, metavar="SECS")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        with RconClient(args.host, args.port, args.password, args.timeout) as rcon:
            response = rcon.command(args.command)
            print(response if response.strip() else "(no response)")
    except RconError as exc:
        print(f"RCON error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
