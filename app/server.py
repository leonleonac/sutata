from __future__ import annotations

import os

from app.main import app


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    if app is None:
        raise SystemExit("FastAPI 未安装，无法启动服务。请先安装 fastapi 和 uvicorn。")

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"uvicorn 未安装，无法启动服务: {exc}") from exc

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = _env_bool("APP_RELOAD", False)
    uvicorn.run("app.server:app", host=host, port=port, reload=reload)
