"""The server CLI must work without importing local model dependencies."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from threading import Thread

import pytest


@pytest.mark.parametrize("no_gen,status", [(False, 200), (True, 200), (False, 503)])
def test_server_cli_reuses_service(no_gen, status):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            payload = ({"retrieved": 1, "results": [{"score": 0.9, "source": "manual", "page": 1,
                        "kind": "text", "text": "边界条件"}]} if no_gen else
                       {"answer": "边界条件", "sources": [], "retrieved": 1})
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Two separate processes, using only the standard library (-S).
        for question in ('温度', '应力'):
            result = subprocess.run([sys.executable, '-S', 'scripts/ask.py', question,
                '--server', f'http://127.0.0.1:{server.server_port}/', '--top-k', '3',
                *(['--no-gen'] if no_gen else [])], capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1], timeout=10,
                env={**os.environ, "no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"})
            if status == 200:
                assert result.returncode == 0, result.stderr
                assert '边界条件' in result.stdout
            else:
                assert result.returncode == 1
                assert '服务请求失败' in result.stderr
                assert 'Traceback' not in result.stderr
        assert requests == [(('/api/search' if no_gen else '/api/query'),
                             {'question': q, 'top_k': 3}) for q in ('温度', '应力')]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
