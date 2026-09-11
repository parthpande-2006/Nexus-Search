import json
import hashlib
import redis
from typing import Optional, List, Dict, Any
from src.config import settings

class RedisCacheManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RedisCacheManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.redis_url = settings.REDIS_URL
        self.client = None
        self.use_in_memory_fallback = False
        self.in_memory_db: Dict[str, bytes] = {}

        try:
            self.client = redis.from_url(self.redis_url, socket_connect_timeout=2)
            self.client.ping()
            print("Connected to Redis successfully.")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"Redis connection failed: {e}. Falling back to in-memory dictionary cache.")
            self.use_in_memory_fallback = True
            self.client = None
        
        self._initialized = True

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[bytes]:
        if self.use_in_memory_fallback:
            return self.in_memory_db.get(key)
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: bytes, expire_seconds: int = 300) -> bool:
        if self.use_in_memory_fallback:
            self.in_memory_db[key] = value
            return True
        try:
            return self.client.set(key, value, ex=expire_seconds)
        except Exception:
            return False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        key = f"embedding:{self._get_hash(text)}"
        cached = self.get(key)
        if cached:
            return json.loads(cached.decode("utf-8"))
        return None

    def set_embedding(self, text: str, embedding: List[float]):
        key = f"embedding:{self._get_hash(text)}"
        serialized = json.dumps(embedding).encode("utf-8")
        self.set(key, serialized, expire_seconds=86400)

    def get_search_results(self, user_id: int, query: str) -> Optional[List[Dict[str, Any]]]:
        key = f"search:{user_id}:{self._get_hash(query)}"
        cached = self.get(key)
        if cached:
            return json.loads(cached.decode("utf-8"))
        return None

    def set_search_results(self, user_id: int, query: str, results: List[Dict[str, Any]], expire_seconds: int = 300):
        key = f"search:{user_id}:{self._get_hash(query)}"
        serialized = json.dumps(results).encode("utf-8")
        self.set(key, serialized, expire_seconds=expire_seconds)

    def get_chat_response(self, user_id: int, query: str) -> Optional[Dict[str, Any]]:
        key = f"chat:{user_id}:{self._get_hash(query)}"
        cached = self.get(key)
        if cached:
            return json.loads(cached.decode("utf-8"))
        return None

    def set_chat_response(self, user_id: int, query: str, response: Dict[str, Any], expire_seconds: int = 300):
        key = f"chat:{user_id}:{self._get_hash(query)}"
        serialized = json.dumps(response).encode("utf-8")
        self.set(key, serialized, expire_seconds=expire_seconds)

    def flush_cache(self):
        if self.use_in_memory_fallback:
            self.in_memory_db.clear()
            return
        try:
            self.client.flushdb()
        except Exception:
            pass
base_cache = RedisCacheManager()
