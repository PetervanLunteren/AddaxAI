"""
No endpoint does database work on the event loop.

FastAPI runs a plain `def` endpoint in a worker thread and an `async def`
endpoint on the event loop itself. A query in an `async def` endpoint
stops every other request, `/health` included, until it returns; on a
large database that froze the dashboard for minutes. So an endpoint that
takes a database session is a plain `def`, with no exceptions: the
project timezone works from a thread (`start_request_timezone`), and
`ws_manager.register_start` hops onto the loop by itself. See "Endpoints
and the event loop" in DEVELOPERS.md.
"""

import inspect

from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from app.db.base import get_db
from app.main import create_app


def _uses_db(dependant: Dependant) -> bool:
    return any(
        dep.call is get_db or _uses_db(dep) for dep in dependant.dependencies
    )


def test_no_async_endpoint_takes_a_database_session():
    offenders = [
        f"{sorted(route.methods)} {route.path} ({route.endpoint.__module__}."
        f"{route.endpoint.__name__})"
        for route in create_app().routes
        if isinstance(route, APIRoute)
        and inspect.iscoroutinefunction(route.endpoint)
        and _uses_db(route.dependant)
    ]
    assert not offenders, (
        "These endpoints are `async def` and take a database session, so "
        "their queries run on the event loop and block every other request. "
        "Make them plain `def` (DEVELOPERS.md, 'Endpoints and the sequence "
        "loop'):\n" + "\n".join(offenders)
    )


def test_the_guard_sees_a_database_session():
    """The check above passes trivially if it cannot find `get_db`."""
    db_routes = [
        route
        for route in create_app().routes
        if isinstance(route, APIRoute) and _uses_db(route.dependant)
    ]
    assert len(db_routes) > 50
