# Usage Guide

## Library Mode

You can import `coreason-synthesis` directly into your Python application to run the pipeline programmatically.

```python
import uuid
from coreason_synthesis.pipeline import SynthesisPipeline
# ... import implementations ...

# Initialize pipeline
pipeline = SynthesisPipeline(...)

# Run synthesis
results = pipeline.run(seeds=[...], config={...}, user_context={...})
```

See the [Home](index.md) page for a complete code example.

## Server Mode (Microservice)

Starting with **v0.2.0**, `coreason-synthesis` can be deployed as a standalone REST API microservice.

### Running with Uvicorn

If you have the package installed in your environment:

```bash
# Set required environment variables
export MCP_BASE_URL="http://mcp-service:8080"
export MCP_API_KEY="your-mcp-key"
# export OPENAI_API_KEY="your-openai-key" # If using real models

# Start the server
uvicorn coreason_synthesis.server:app --host 0.0.0.0 --port 8000
```

### Running with Docker

The Docker image is pre-configured to run the server.

```bash
docker run -p 8000:8000 \
  -e MCP_BASE_URL="http://host.docker.internal:8080" \
  coreason/synthesis:0.2.0
```

### API Endpoints

#### `POST /synthesis/run`

Execute the synthesis pipeline.

**Request Body:**

```json
{
  "seeds": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "context": "Patient has Stage IV Lung Cancer...",
      "question": "Is this patient eligible?",
      "expected_output": "Yes, based on criteria X...",
      "provenance": "MANUAL_SEED"
    }
  ],
  "config": {
    "target_count": 10,
    "perturbation_rate": 0.2
  },
  "user_context": {
    "user_id": "user_123"
  }
}
```

**Response:**

A list of `SyntheticTestCase` objects.

```json
[
  {
    "verbatim_context": "...",
    "synthetic_question": "...",
    "complexity": 8.5,
    "provenance": "VERBATIM_SOURCE",
    ...
  }
]
```

#### `GET /health`

Check service status.

**Response:**

```json
{
  "status": "active",
  "components": "ready"
}
```
