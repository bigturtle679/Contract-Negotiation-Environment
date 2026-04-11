"""
Client for the Contract Negotiation Environment.

Provides a simple HTTP client for programmatic interaction with the
contract negotiation server running at a given base URL.

If openenv-core is installed, also provides an EnvClient subclass.
"""
from __future__ import annotations

from typing import Optional

# Use correct model names (aliases defined in contract_env/__init__.py)
from contract_env.env.models import Action, Observation

ContractAction = Action
ContractObservation = Observation

# EnvClient from openenv-core is optional — only available on machines
# where openenv-core is installed (e.g., during openenv validate).
try:
    from openenv.core.env_client import EnvClient as _EnvClient
    _OPENENV_AVAILABLE = True
except ImportError:
    _EnvClient = object  # type: ignore[assignment,misc]
    _OPENENV_AVAILABLE = False


class ContractNegotiationClient(_EnvClient):  # type: ignore[misc]
    """HTTP client for the Contract Negotiation Environment server.

    Usage (with openenv-core installed):
        async with ContractNegotiationClient(base_url="https://your-space.hf.space") as client:
            result = await client.reset()
            result = await client.step(Action(action_type="FLAG_RISK"))

    Simple HTTP usage (no openenv-core required):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:7860") as http:
            obs = (await http.post("/reset")).json()
            step = (await http.post("/step", json={"action_type": "FLAG_RISK"})).json()
    """

    DOCKER_IMAGE = "contract-negotiation-env"
    ENV_NAME = "contract_negotiation"

    def __init__(
        self,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        if _OPENENV_AVAILABLE:
            super().__init__(
                base_url=base_url or "http://localhost:7860",
                action_type=ContractAction,
                observation_type=ContractObservation,
                **kwargs,
            )
        else:
            self.base_url = base_url or "http://localhost:7860"

    @classmethod
    async def from_docker_image(
        cls,
        image_name: Optional[str] = None,
        port: int = 7860,
        **kwargs,
    ) -> "ContractNegotiationClient":
        """Create a client pointing at a locally running Docker container.

        Args:
            image_name: Docker image name. Defaults to DOCKER_IMAGE.
            port: Host port to connect to. Defaults to 7860.
        """
        return cls(base_url=f"http://localhost:{port}", **kwargs)
