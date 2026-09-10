"""Shared TigerGraph connection helper for all pipelines.

Savanna auth flow used here (verified against a live Savanna 4.2.5 workspace):
the value you get from GraphStudio ("CREATE SECRET" / the workspace Connect
panel) is a *database secret*, not a REST++ token. It must be exchanged for a
short-lived JWT via `POST {host}/gsql/v1/tokens` with a JSON body that
includes the target graph name -- pyTigerGraph 2.0.4's own getToken() call
does not include the graph name in that request and fails against Savanna,
so we do the exchange manually here and hand the resulting JWT to
TigerGraphConnection as apiToken.
"""
import os
import time
import requests
from dotenv import load_dotenv
import pyTigerGraph as tg

load_dotenv()

DEFAULT_TOKEN_LIFETIME_SEC = 2_592_000  # 30 days, matches TigerGraph's default


def _fetch_jwt(host: str, graph: str, secret: str, lifetime: int = DEFAULT_TOKEN_LIFETIME_SEC) -> str:
    url = host.rstrip("/") + "/gsql/v1/tokens"
    body = {"secret": secret, "lifetime": lifetime, "graph": graph}
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"TigerGraph token request failed: {data}")
    return data["token"]


def get_connection(graphname: str = None) -> tg.TigerGraphConnection:
    """Build an authenticated TigerGraphConnection against Savanna.

    Reads TG_HOST, TG_GRAPHNAME, TG_SECRET from the environment (.env),
    exchanges the secret for a fresh JWT, and returns a ready connection.
    """
    host = os.environ["TG_HOST"]
    graph = graphname or os.environ.get("TG_GRAPHNAME", "test")
    secret = os.environ["TG_SECRET"]

    jwt = _fetch_jwt(host, graph, secret)
    conn = tg.TigerGraphConnection(host=host, graphname=graph, apiToken=jwt)
    return conn


if __name__ == "__main__":
    t0 = time.time()
    conn = get_connection()
    print("echo:", conn.echo())
    print("graph:", conn.graphname)
    print("vertex types:", conn.getVertexTypes())
    print("edge types:", conn.getEdgeTypes())
    print(f"connected in {time.time() - t0:.2f}s")
