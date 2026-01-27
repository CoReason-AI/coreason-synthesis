import pytest
from fastapi.testclient import TestClient
import respx
from httpx import Response

from coreason_synthesis.server import app
from coreason_synthesis.models import SeedCase

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "active", "components": "ready"}

def test_run_synthesis_mocked():
    # Mock MCP response using respx
    with respx.mock(base_url="http://localhost:8080") as respx_mock:
        respx_mock.post("/search").mock(return_value=Response(200, json={
            "results": [
                {
                    "content": "This is a test document. It has enough length to be considered a valid chunk for the extractor.",
                    "source_urn": "urn:test:doc1",
                    "metadata": {"page_number": 1}
                }
            ]
        }))

        with TestClient(app) as client:
            seeds = [
                {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "context": "Context",
                    "question": "Question?",
                    "expected_output": {"answer": "Answer"}
                }
            ]
            payload = {
                "seeds": seeds,
                "config": {"target_count": 1},
                "user_context": {"user_id": "test"}
            }

            response = client.post("/synthesis/run", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            # We expect at least one case generated because we have a valid doc and MockTeacher
            assert len(data) > 0
            first_case = data[0]
            assert "synthetic_question" in first_case
            assert first_case["provenance"] == "VERBATIM_SOURCE"
