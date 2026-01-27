# src/coreason_synthesis/server.py

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from coreason_synthesis.analyzer import PatternAnalyzerImpl
from coreason_synthesis.appraiser import AppraiserImpl
from coreason_synthesis.clients.mcp import HttpMCPClient
from coreason_synthesis.compositor import CompositorImpl
from coreason_synthesis.extractor import ExtractorImpl
from coreason_synthesis.forager import ForagerImpl
from coreason_synthesis.mocks.embedding import DummyEmbeddingService
from coreason_synthesis.mocks.teacher import MockTeacher
from coreason_synthesis.models import SeedCase, SyntheticTestCase
from coreason_synthesis.perturbator import PerturbatorImpl
from coreason_synthesis.pipeline import SynthesisPipelineAsync


class RunRequest(BaseModel):
    seeds: List[SeedCase]
    config: Dict[str, Any]
    user_context: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration
    # Default to localhost for dev/test if not set, though production should set it.
    mcp_base_url = os.getenv("MCP_BASE_URL", "http://localhost:8080")
    mcp_api_key = os.getenv("MCP_API_KEY")

    # Not used by mocks, but noted for future real implementation
    # openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize Components

    # Clients
    mcp_client = HttpMCPClient(base_url=mcp_base_url, api_key=mcp_api_key)

    # Services (Using Mocks as real implementations are not available in this scope)
    teacher = MockTeacher()
    embedder = DummyEmbeddingService()

    # Pipeline Components
    analyzer = PatternAnalyzerImpl(teacher=teacher, embedder=embedder)
    forager = ForagerImpl(mcp_client=mcp_client, embedder=embedder)
    extractor = ExtractorImpl()
    compositor = CompositorImpl(teacher=teacher)
    perturbator = PerturbatorImpl()
    appraiser = AppraiserImpl(teacher=teacher, embedder=embedder)

    # Pipeline
    pipeline = SynthesisPipelineAsync(
        analyzer=analyzer,
        forager=forager,
        extractor=extractor,
        compositor=compositor,
        perturbator=perturbator,
        appraiser=appraiser
    )

    # Store in app state and enter context
    app.state.pipeline = pipeline
    await pipeline.__aenter__()

    yield

    # Cleanup
    await pipeline.__aexit__(None, None, None)
    await mcp_client.close()


app = FastAPI(lifespan=lifespan)


@app.post("/synthesis/run", response_model=List[SyntheticTestCase])
async def run_synthesis(request: RunRequest):
    pipeline: SynthesisPipelineAsync = app.state.pipeline
    results = await pipeline.run(
        seeds=request.seeds,
        config=request.config,
        user_context=request.user_context
    )
    return results


@app.get("/health")
async def health_check():
    return {"status": "active", "components": "ready"}
