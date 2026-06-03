from neo4j import GraphDatabase, Driver
from typing import Optional, List, Dict, Any
from loguru import logger
from app.config import get_settings

settings = get_settings()
_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            _driver.verify_connectivity()
            logger.info(f"✅ Connected to Neo4j at {settings.neo4j_uri}")
        except Exception as e:
            logger.warning(f"⚠️  Neo4j not available: {e}. Using in-memory fallback.")
            _driver = None
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


class Neo4jSession:
    """Context manager for Neo4j sessions with fallback."""

    def __init__(self):
        self.driver = get_driver()
        self.session = None

    def __enter__(self):
        if self.driver:
            self.session = self.driver.session()
        return self

    def __exit__(self, *args):
        if self.session:
            self.session.close()

    def run(self, query: str, **params) -> List[Dict[str, Any]]:
        if not self.session:
            logger.warning("Neo4j unavailable — returning empty results")
            return []
        result = self.session.run(query, **params)
        return [dict(record) for record in result]

    def run_write(self, query: str, **params) -> List[Dict[str, Any]]:
        """Run a write transaction."""
        if not self.session:
            return []
        with self.driver.session() as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, **params))
            )
            return [dict(r) for r in result]
