# Requirements

## System Requirements

- **Python**: 3.12, 3.13, or 3.14

## Python Dependencies

The core library depends on the following packages:

- **anyio** (>=4.12.1): Asynchronous I/O support.
- **httpx** (>=0.28.1): Async HTTP client.
- **numpy** (>=2.4.1): Numerical operations.
- **pydantic** (>=2.12.5): Data validation and settings management.
- **loguru** (>=0.7.2): Robust logging.
- **requests** (>=2.32.5): Synchronous HTTP client (for Foundry integration).
- **aiofiles** (>=25.1.0): File I/O support.

### Server Mode (Microservice)

To run `coreason-synthesis` as a microservice (Server Mode), the following additional dependencies are required (included by default in v0.2.0+):

- **fastapi**: High-performance web framework.
- **uvicorn[standard]**: ASGI server implementation.

## Development Dependencies

For contributing to the project:

- **pytest** / **pytest-cov** / **pytest-asyncio**: Testing framework.
- **ruff**: Linting and formatting.
- **mypy**: Static type checking.
- **pre-commit**: Git hook management.
- **mkdocs-material**: Documentation generation.
