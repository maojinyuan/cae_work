import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_cli_routes_generation_without_loading_pipeline(monkeypatch, capsys):
    from core import local_service
    ask = importlib.import_module('scripts.ask')
    query = Mock(return_value={'answer': '答案', 'sources': []})
    monkeypatch.setattr(local_service, 'ask', query)
    monkeypatch.setitem(sys.modules, 'knowledge.rag', None)
    monkeypatch.setattr(sys, 'argv', ['ask.py', '热边界条件', '--top-k', '3'])
    assert ask.main() == 0
    query.assert_called_once_with('热边界条件', 3)
    assert '答案' in capsys.readouterr().out


def test_no_gen_does_not_start_service(monkeypatch):
    from core import local_service
    ask = importlib.import_module('scripts.ask')
    query = Mock(side_effect=AssertionError('must not start service'))
    search = Mock(return_value=[])
    monkeypatch.setattr(local_service, 'ask', query)
    monkeypatch.setitem(sys.modules, 'knowledge.rag', SimpleNamespace(
        get_pipeline=lambda: SimpleNamespace(search=search)))
    monkeypatch.setattr(sys, 'argv', ['ask.py', '温度', '--no-gen'])
    assert ask.main() == 0
    search.assert_called_once_with('温度', None)
    query.assert_not_called()


def test_start_failure_is_reported(monkeypatch, tmp_path):
    from core import local_service as service
    monkeypatch.setattr(service.config, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(service, '_ready', lambda: False)
    monkeypatch.setattr(service, '_listening', lambda: False)
    monkeypatch.setattr(service, '_spawn', lambda: SimpleNamespace(poll=lambda: 1))
    with pytest.raises(RuntimeError, match='启动失败'):
        service.ensure_started()


def test_start_timeout_does_not_kill_loading_service(monkeypatch, tmp_path):
    from core import local_service as service
    monkeypatch.setattr(service.config, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(service.config, 'SERVICE_START_TIMEOUT', 0.01)
    monkeypatch.setattr(service, '_ready', lambda: False)
    monkeypatch.setattr(service, '_listening', lambda: True)
    spawn = Mock(side_effect=AssertionError('already listening'))
    monkeypatch.setattr(service, '_spawn', spawn)
    with pytest.raises(RuntimeError, match='超时'):
        service.ensure_started()
    spawn.assert_not_called()


def test_background_lifecycle_with_real_http(monkeypatch, tmp_path):
    """Real server processes and HTTP; only expensive model loading is replaced."""
    import concurrent.futures
    import os
    import socket
    import subprocess
    import time
    from urllib.error import HTTPError
    from core import local_service as service

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    monkeypatch.setattr(service.config, 'SERVICE_PORT', port)
    monkeypatch.setattr(service.config, 'SERVICE_START_TIMEOUT', 10)
    monkeypatch.setattr(service.config, 'CACHE_DIR', tmp_path)
    counter = tmp_path / 'loads.txt'
    runner = tmp_path / 'fake_runner.py'
    runner.write_text('''
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
sys.path.insert(0, os.environ['CAE_TEST_PROJECT'])
from core import local_service
pipeline = SimpleNamespace(llm=None)
def load():
    with Path(os.environ['LOAD_COUNTER']).open('a') as f:
        f.write(str(os.getpid()) + '\\n')
    time.sleep(0.3)
    return object()
def ask(question, top_k):
    if question == 'slow':
        Path(os.environ['LOAD_COUNTER'] + '.active').touch()
        time.sleep(1)
    return {'answer': question, 'sources': [], 'pid': os.getpid()}
pipeline.ask = ask
rag = ModuleType('knowledge.rag')
rag.get_pipeline = lambda: pipeline
llm = ModuleType('models.llm')
llm.get_llm = load
sys.modules['knowledge.rag'] = rag
sys.modules['models.llm'] = llm
''')
    (tmp_path / 'sitecustomize.py').write_text(
        "import sys\nif sys.argv[0].endswith('serve.py'):\n    import fake_runner\n")
    processes = []
    env = {**os.environ, 'CAE_SERVICE_PORT': str(port), 'CAE_CACHE_DIR': str(tmp_path),
           'CAE_STORAGE_DIR': str(service.config.STORAGE_DIR), 'LOAD_COUNTER': str(counter),
           'CAE_TEST_PROJECT': str(service.config.PROJECT_ROOT), 'PYTHONPATH': str(tmp_path)}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    original_spawn = service._spawn

    def spawn():
        process = original_spawn()
        processes.append(process)
        return process

    monkeypatch.setattr(service, '_spawn', spawn)
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            answers = list(pool.map(service.ask, ['热边界', '约束']))
        assert answers[0]['pid'] == answers[1]['pid']
        assert len(counter.read_text().splitlines()) == 1
        # The unmodified CLI command connects without importing model packages.
        for question in ('温度', '应力'):
            response = subprocess.run([sys.executable, 'scripts/ask.py', question], env=env,
                capture_output=True, text=True, cwd=service.config.PROJECT_ROOT, timeout=10)
            assert response.returncode == 0, response.stderr
            assert question in response.stdout
        assert len(counter.read_text().splitlines()) == 1
        with pytest.raises(HTTPError) as denied:
            service._request('/api/runtime/stop', {})
        assert denied.value.code == 403
        assert service._ready()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            active = pool.submit(service.ask, 'slow')
            deadline = time.monotonic() + 5
            while not counter.with_suffix('.txt.active').exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert '已停止' in service.stop()
            assert active.done(), 'stop must wait for the active generation'
            assert active.result()['answer'] == 'slow'
        assert '未运行' in service.stop()
        answer = service.ask('重启后')
        assert answer['pid'] != answers[0]['pid']
        assert len(counter.read_text().splitlines()) == 2
        assert '已停止' in service.stop()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


def test_foreign_service_is_not_reused_or_stopped(monkeypatch):
    from core import local_service as service
    monkeypatch.setattr(service, '_request', lambda *a, **kw: {'service': 'other'})
    spawn = Mock()
    monkeypatch.setattr(service, '_spawn', spawn)
    with pytest.raises(RuntimeError, match='其他服务'):
        service.ensure_started()
    with pytest.raises(RuntimeError, match='其他服务'):
        service.stop()
    spawn.assert_not_called()
