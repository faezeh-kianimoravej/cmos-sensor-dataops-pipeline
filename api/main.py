"""Main entry point for the API."""

from api.app import app
from src.observed.utils.config import load_config

if __name__ == "__main__":
    import uvicorn

    # Load config for API settings
    config = load_config("configs/default.yaml")
    api_config = config.get("api", {})

    port = api_config.get("port", 8000)
    reload = api_config.get("reload", False)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info",
    )
