def get_store():
    from .chroma import ChromaStore
    # Keep a single process-local store instance.
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store

_store = None

__all__ = ["get_store"]
