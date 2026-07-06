"""
Rate limiter instance shared between app.main (which wires it into the
FastAPI app/exception handler) and route modules (which apply per-route
limits) — kept in its own module to avoid a circular import between them.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])
