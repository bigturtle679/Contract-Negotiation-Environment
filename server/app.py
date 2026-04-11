"""Re-export the FastAPI app for multi-mode deployment compatibility.

The openenv validator expects ``server/app.py`` at the repository root.
The canonical implementation lives in ``contract_env.server.app``.
"""

from contract_env.server.app import app  # noqa: F401
