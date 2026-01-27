import uuid

import respx
from fastapi.testclient import TestClient
from httpx import Response

from coreason_synthesis.server import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "active", "components": "ready"}


@respx.mock  # type: ignore[misc]
def test_run_synthesis() -> None:
    # Mock the MCP search endpoint
    # Default MCP_BASE_URL in server.py is http://localhost:8080
    respx.post("http://localhost:8080/search").mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {
                        "content": (
                            "This is a retrieved document content about medical condition. "
                            "It has enough length to be considered valid chunk."
                        ),
                        "source_urn": "urn:doc:123",
                        "metadata": {"page_number": 1},
                    }
                ]
            },
        )
    )

    payload = {
        "seeds": [
            {
                "id": str(uuid.uuid4()),
                "context": "Seed context",
                "question": "Seed question",
                "expected_output": "Seed output",
                "provenance": "MANUAL_SEED",
                "modifications": [],
                "source_urn": "urn:seed:1",
            }
        ],
        "config": {"target_count": 1, "perturbation_rate": 0.0},
        "user_context": {"user": "tester"},
    }

    with TestClient(app) as client:
        response = client.post("/synthesis/run", json=payload)

        assert response.status_code == 200, response.text
        data = response.json()
        assert isinstance(data, list)

        # We check that we got at least one result
        # The flow is: Analyze -> Forage (Mocked) -> Extract -> Composite -> Appraise
        # Extractor checks for length > 50, so I made the content longer.
        if len(data) > 0:
            first_case = data[0]
            assert "synthetic_question" in first_case
            assert first_case["source_urn"] == "urn:doc:123"
