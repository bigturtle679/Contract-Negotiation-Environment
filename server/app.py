"""Re-export the FastAPI app for multi-mode deployment compatibility.

The openenv validator expects a ``def main():`` function defined in this file.
The canonical app lives in ``contract_env.server.app``.
"""
import os

from contract_env.server.app import app  # noqa: F401


def main() -> None:
    import uvicorn

    uvicorn.run(
        "contract_env.server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
    )


if __name__ == "__main__":
    main()
