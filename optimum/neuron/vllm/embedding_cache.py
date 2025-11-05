import threading

# Global cache for embeddings with thread safety
# This is necessary because vLLM's architecture separates the model runner
# (which computes embeddings) from the IO processor (which formats output).
# There's no direct communication channel between them, so we use a global
# cache as a bridge.
_embedding_cache = {}
_cache_lock = threading.Lock()

def store_embeddings(request_ids, embeddings):
    """Store embeddings for given request IDs (thread-safe)."""
    with _cache_lock:
        for req_id, emb in zip(request_ids, embeddings):
            _embedding_cache[req_id] = emb

def get_embedding(request_id):
    """Retrieve embedding for a request ID (thread-safe)."""
    with _cache_lock:
        return _embedding_cache.pop(request_id, None)

def clear_cache():
    """Clear all cached embeddings (thread-safe)."""
    with _cache_lock:
        _embedding_cache.clear()
