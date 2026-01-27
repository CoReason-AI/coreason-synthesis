import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

from coreason_identity.models import UserContext
from fastapi import FastAPI
from pydantic import BaseModel

from .analyzer import PatternAnalyzerImpl
from .appraiser import AppraiserImpl
from .clients.mcp import HttpMCPClient
from .compositor import CompositorImpl
from .extractor import ExtractorImpl
from .forager import ForagerImpl
from .mocks.embedding import DummyEmbeddingService
from .mocks.teacher import MockTeacher
from .models import SeedCase, SyntheticTestCase
from .perturbator import PerturbatorImpl
from .pipeline import SynthesisPipelineAsync


# Define the request model
class SynthesisRunRequest(BaseModel):
    seeds: List[SeedCase]
    config: Dict[str, Any]
    user_context: UserContext


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager to initialize the synthesis pipeline and its components.
    ensures that resources (like HTTP clients) are properly closed on shutdown.
    """
    # 1. Initialize Clients
    # We retrieve configuration from environment variables
    mcp_base_url = os.environ.get("MCP_BASE_URL", "http://localhost:8080")
    mcp_api_key = os.environ.get("MCP_API_KEY")

    # Initialize the MCP Client
    # This client will be used by the Forager to search for documents
    mcp_client = HttpMCPClient(base_url=mcp_base_url, api_key=mcp_api_key)

    # Initialize Teacher and Embedder
    # Note: In a production deployment, these would use real clients (e.g., OpenAI, Azure).
    # Since only Mocks are available in this repository, we use them here.
    # We check for OPENAI_API_KEY to simulate reading config, even if unused by the mock.
    _ = os.environ.get("OPENAI_API_KEY")

    teacher = MockTeacher()
    embedder = DummyEmbeddingService()

    # 2. Assemble Components
    analyzer = PatternAnalyzerImpl(teacher, embedder)
    forager = ForagerImpl(mcp_client, embedder)
    extractor = ExtractorImpl()
    compositor = CompositorImpl(teacher)
    perturbator = PerturbatorImpl()
    appraiser = AppraiserImpl(teacher, embedder)

    # 3. Initialize Pipeline
    # We use the pipeline as an async context manager to ensure its internal client is closed
    pipeline = SynthesisPipelineAsync(
        analyzer=analyzer,
        forager=forager,
        extractor=extractor,
        compositor=compositor,
        perturbator=perturbator,
        appraiser=appraiser,
    )

    async with pipeline:
        # Store pipeline in app state for access in endpoints
        app.state.pipeline = pipeline
        yield

    # Close the MCP client explicitly as it was created outside the pipeline
    await mcp_client.close()


app: FastAPI = FastAPI(lifespan=lifespan)


@app.post("/synthesis/run", response_model=List[SyntheticTestCase])  # type: ignore[misc]
async def run_synthesis(request: SynthesisRunRequest) -> List[SyntheticTestCase]:
    """
    Executes the synthesis pipeline.

    Accepts seeds, config, and user context, and returns a list of appraised synthetic test cases.
    """
    pipeline: SynthesisPipelineAsync = app.state.pipeline

    results = await pipeline.run(seeds=request.seeds, config=request.config, user_context=request.user_context)

    return results


@app.get("/health")  # type: ignore[misc]
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "active", "components": "ready"}
