"""
Creates a cache instance which stores and retieves data frames
for the application
"""

# Default to use Feather files for data caching rather than redis
use_redis = False

if use_redis:
    import os
    from flask_caching import Cache

    CACHE_CONFIG = {
        "CACHE_TYPE": "redis",
        "CACHE_REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6379"),
    }
    cache = Cache()
    cache.init_app(app.server, config=CACHE_CONFIG)
else:
    from .feather_cache import FeatherCache
    cache = FeatherCache()
