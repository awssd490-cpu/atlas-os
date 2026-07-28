"""SQL connection protocol definition.

Re-exported from ``app.storage.interfaces`` for import convenience.
This module exists so connection implementations can import the protocol
without pulling in the full interface file.
"""

from app.storage.interfaces import SQLConnection, Row

__all__ = [
    "SQLConnection",
    "Row",
]
