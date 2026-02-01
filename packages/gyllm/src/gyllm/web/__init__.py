from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any) -> Any:
    from gyllm.web.server import create_app as _create_app

    return _create_app(*args, **kwargs)
