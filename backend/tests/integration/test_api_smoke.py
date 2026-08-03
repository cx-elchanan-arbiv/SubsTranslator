"""CORS preflight contract for the browser-facing endpoints.

Driven through the Flask test client on purpose. The previous version forked a second
copy of the app with ``os.execv`` and polled ``127.0.0.1:8081`` — inside the container
that port already belongs to the running backend, so the fixture's health check passed
against the *existing* server while the forked one failed to bind, and the tests silently
measured something other than the code under test.
"""

import pytest

ALLOWED_ORIGIN = "http://localhost:3000"


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.mark.integration
def test_cors_preflight_is_answered_for_the_dev_origin(client):
    """The browser must get a usable preflight answer before it will POST /youtube.

    Note the header is the echoed origin, not a literal ``*``: the app sets
    ``supports_credentials``, and the CORS spec forbids the wildcard on credentialed
    requests. An earlier version of this test asserted ``== "*"`` and had been failing
    ever since credentials were turned on.
    """
    response = client.options(
        "/youtube",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code in (200, 204)
    assert response.headers.get("Access-Control-Allow-Origin") == ALLOWED_ORIGIN
    assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")


@pytest.mark.integration
def test_a_credentialed_preflight_never_answers_with_a_wildcard(client):
    """A wildcard plus credentials is rejected by every browser — belt and braces."""
    response = client.options(
        "/youtube",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("Access-Control-Allow-Origin") != "*"
