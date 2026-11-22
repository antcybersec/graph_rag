"""TigerGraph connection for Round 2, against either backend.

Two deployments have to work from identical code, because they serve different
purposes in this submission:

  local       TigerGraph Community Edition 4.2.6 in Docker. No credits, no
              workspace to resume, and a judge can rebuild the whole graph with
              no cloud account at all -- which is what "clean setup" and "live
              reproduction" in the repeatability criterion actually reward.
  savanna     A hosted workspace, for latency figures measured on real
              hardware rather than on x86 emulation over Apple silicon.

The auth paths are genuinely different and neither is interchangeable. Savanna
issues a *database secret* that must be exchanged for a short-lived JWT via
POST /gsql/v1/tokens with the graph name in the body -- pyTigerGraph's own
getToken() omits the graph name and fails against Savanna, which Round 1
established the hard way. Community Edition ships with username/password auth
and no token exchange at all.

So the mode is explicit (TG_MODE), inferred from the host only as a
convenience, and `src/common/tg_conn.py` is left exactly as Round 1 verified
it rather than being generalised in place.
"""
from __future__ import annotations

import os

import pyTigerGraph as tg
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TOKEN_LIFETIME_SEC = 2_592_000  # 30 days, TigerGraph's own default


def _mode(host: str) -> str:
    explicit = (os.environ.get("TG_MODE") or "").strip().lower()
    if explicit in ("local", "savanna"):
        return explicit
    return "local" if ("localhost" in host or "127.0.0.1" in host) else "savanna"


def _savanna_jwt(host: str, graph: str, secret: str) -> str:
    resp = requests.post(host.rstrip("/") + "/gsql/v1/tokens", timeout=30,
                         json={"secret": secret, "graph": graph,
                               "lifetime": DEFAULT_TOKEN_LIFETIME_SEC})
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"TigerGraph token request failed: {data}")
    return data["token"]


def get_connection(graphname: str | None = None):
    """An authenticated connection, whichever backend is configured."""
    host = os.environ.get("TG_HOST", "http://localhost")
    graph = graphname or os.environ.get("TG_GRAPHNAME", "r2")
    mode = _mode(host)

    if mode == "local":
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graph,
            username=os.environ.get("TG_USERNAME", "tigergraph"),
            password=os.environ.get("TG_PASSWORD", "tigergraph"),
            restppPort=os.environ.get("TG_RESTPP_PORT", "9000"),
            gsPort=os.environ.get("TG_GS_PORT", "14240"),
        )
        # Community Edition may or may not have token auth enabled depending on
        # how it was provisioned. Ask for a token, and carry on without one if
        # the server does not require it -- failing here would block a build
        # that is perfectly able to proceed.
        try:
            conn.getToken(conn.createSecret())
        except Exception:
            pass
        return conn

    secret = os.environ["TG_SECRET"]
    return tg.TigerGraphConnection(host=host, graphname=graph,
                                   apiToken=_savanna_jwt(host, graph, secret))


def describe() -> str:
    host = os.environ.get("TG_HOST", "http://localhost")
    return f"{_mode(host)} @ {host} graph={os.environ.get('TG_GRAPHNAME', 'r2')}"
