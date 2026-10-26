"""
One command to start everything:      python run.py
Opens the web app in your browser.  Uses FastAPI + uvicorn if they are installed, otherwise a built-in server (no installs needed).
Options:  --port 8000   --no-browser   --stdlib (force the built-in server)
"""
import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def free_port(start):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def prepare_models():
    """The zip ships without the large classical-model files (they are rebuilt deterministically in 1-3 minutes per dataset, once)."""
    from src import zoo
    for d in sorted((Path(__file__).resolve().parent / "artifacts" / "zoo").glob("FD*")):
        try:
            zoo.ensure_classical(Path(__file__).resolve().parent, d.name, log=lambda m: print("  first start:", m, flush=True))
        except Exception as exc:                      # the dashboard still works from the pre-computed results
            print(f"  note: could not prepare live models for {d.name} ({exc}); live prediction will be unavailable for it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--stdlib", action="store_true")
    a = ap.parse_args()
    from src import api_core
    if not api_core.available():
        sys.exit("No model zoo found in artifacts/zoo/.  Run the notebooks 01 -> 02 -> 03 first (or: python tools/build_zoo.py).")
    prepare_models()
    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}"
    use_fastapi = False
    if not a.stdlib:
        try:
            import fastapi, uvicorn  # noqa: F401
            use_fastapi = True
        except ImportError:
            pass
    print(f"\n  Predictive Maintenance & RUL  ->  {url}   ({'FastAPI + uvicorn' if use_fastapi else 'built-in server'})   Ctrl+C to stop\n")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    if use_fastapi:
        import uvicorn
        uvicorn.run("backend.main:app", host="127.0.0.1", port=port, log_level="warning")
    else:
        from src.server_std import make_server
        srv = make_server("127.0.0.1", port)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
