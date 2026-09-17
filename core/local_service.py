"""Local CLI service lifecycle. Model imports belong only in the server process."""
from __future__ import annotations

import json
import errno
import os
import secrets
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from core import config


def _path(suffix):
    return config.CACHE_DIR / f"ask-service-{config.SERVICE_PORT}.{suffix}"


def _identity():
    return {"service": "cae-ask-v1", "project": str(config.PROJECT_ROOT),
            "storage": str(config.STORAGE_DIR)}


def _request(path, payload=None, timeout=1, headers=None):
    request = Request(
        f"http://127.0.0.1:{config.SERVICE_PORT}{path}",
        data=None if payload is None else json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    # A local model request must not travel through an environment HTTP proxy.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def _ready():
    try:
        status = _request('/api/runtime', timeout=0.5)
    except HTTPError as exc:
        raise RuntimeError(f"端口 {config.SERVICE_PORT} 上不是兼容的后台服务，请修改 CAE_SERVICE_PORT。") from exc
    except (URLError, OSError):
        return False
    except ValueError as exc:
        raise RuntimeError('后台服务返回了无效响应，请检查 CAE_SERVICE_PORT。') from exc
    if not isinstance(status, dict) or any(status.get(k) != v for k, v in _identity().items()):
        raise RuntimeError('该端口属于其他服务或其他项目，请修改 CAE_SERVICE_PORT。')
    return True


def _listening():
    try:
        with socket.create_connection(('127.0.0.1', config.SERVICE_PORT), timeout=0.2):
            return True
    except OSError:
        return False


def _spawn():
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    options = ({'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == 'nt' else {'start_new_session': True})
    with _path('log').open('ab') as log:
        return subprocess.Popen(
            [sys.executable, str(config.PROJECT_ROOT / 'scripts' / 'serve.py')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, **options,
        )


def ensure_started():
    if _ready():
        return
    print(f"正在等待后台模型服务（首次启动需加载权重），日志：{_path('log')}", file=sys.stderr)
    process = None if _listening() else _spawn()
    deadline = time.monotonic() + config.SERVICE_START_TIMEOUT
    while time.monotonic() < deadline:
        if _ready():
            return
        if process is not None and process.poll() is not None and not _listening():
            raise RuntimeError(f"后台服务启动失败，请查看日志：{_path('log')}")
        time.sleep(0.2)
    raise RuntimeError(f"等待后台服务超时；加载可能仍在进行，请查看 {_path('log')}，稍后重试。")


def ask(question, top_k=None):
    ensure_started()
    try:
        return _request('/api/query', {'question': question, 'top_k': top_k},
                        timeout=config.SERVICE_REQUEST_TIMEOUT)
    except (URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"问答请求失败（不会自动重复生成）：{exc}；日志：{_path('log')}") from exc


def stop():
    if not _ready():
        if _listening():
            raise RuntimeError(f"服务尚未就绪，请等待加载完成再停止；日志：{_path('log')}")
        return '后台服务未运行。'
    token = _path('token').read_text(encoding='utf-8')
    _request('/api/runtime/stop', {}, headers={'X-CAE-Token': token})
    deadline = time.monotonic() + 30
    while _listening():
        if time.monotonic() >= deadline:
            return '已请求停止；服务仍在等待当前请求结束，请稍后检查。'
        time.sleep(0.2)
    return '后台服务已停止，模型内存/显存已释放。'


def serve():
    # Reserve the port BEFORE importing/loading models. Concurrent starters lose
    # the bind race and exit without allocating another copy of the weights.
    with socket.socket() as listener:
        if os.name == 'nt':
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', config.SERVICE_PORT))
            listener.listen(128)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return  # Another starter won; its listening socket owns the port.
            raise
        import uvicorn
        from fastapi import Header, HTTPException
        from app.main import app
        from knowledge.rag import get_pipeline
        from models.llm import get_llm

        print(f'正在加载 Embedding、知识库和 LLM… PID={os.getpid()}', flush=True)
        pipeline = get_pipeline()
        if pipeline.llm is None:
            pipeline.llm = get_llm()
        token = secrets.token_urlsafe(32)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(_path('token'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(token)
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=config.SERVICE_PORT, workers=1))

        @app.get('/api/runtime')
        def runtime():
            return {**_identity(), 'pid': os.getpid()}

        @app.post('/api/runtime/stop')
        def shutdown(x_cae_token: str = Header(default='')):
            if not secrets.compare_digest(x_cae_token, token):
                raise HTTPException(403, 'Invalid service token')
            server.should_exit = True
            return {'status': 'stopping'}

        print('模型加载完成，服务即将就绪。', flush=True)
        # Uvicorn closes its socket before draining active requests. Keep our
        # reference open until draining completes so a new model cannot start.
        with listener.dup() as http_listener:
            server.run(sockets=[http_listener])
