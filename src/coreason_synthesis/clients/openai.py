# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_synthesis

import os
from typing import Any, Optional, Type

import requests

from coreason_synthesis.interfaces import T, TeacherModel
from coreason_synthesis.utils.http import create_retry_session
from coreason_synthesis.utils.logger import logger


class OpenAITeacher(TeacherModel):
    """
    Concrete implementation of the TeacherModel using OpenAI-compatible API.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set. API calls may fail.")

        # Use shared retry session logic
        self.session = create_retry_session(api_key=self.api_key, max_retries=max_retries)

        # Override Authorization header format for OpenAI (Bearer)
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generates text based on a prompt and optional context using chat completions.
        """
        messages = self._build_messages(prompt, context)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,  # Low temp for more deterministic output
        }

        try:
            response = self._post_completion(payload)
            content = response["choices"][0]["message"]["content"]
            return str(content)
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}") from e

    def generate_structured(self, prompt: str, response_model: Type[T], context: Optional[str] = None) -> T:
        """
        Generates a structured object using OpenAI's structured outputs (if supported) or JSON mode.
        For broad compatibility, we'll use JSON mode and Pydantic validation.
        """
        messages = self._build_messages(prompt, context)

        # Append instruction to return JSON matching the schema
        schema = response_model.model_json_schema()
        system_instruction = (
            f"You must respond with valid JSON that matches the following schema:\n{schema}\n"
            "Do not include any markdown formatting (like ```json)."
        )

        # Inject system instruction if not present, or append to existing system message?
        # Simpler to append to user message or add a system message.
        messages.insert(0, {"role": "system", "content": system_instruction})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        try:
            response = self._post_completion(payload)
            content = response["choices"][0]["message"]["content"]

            # Validate and parse with Pydantic
            return response_model.model_validate_json(content)

        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            logger.error(f"OpenAI structured generation failed: {e}")
            raise RuntimeError(f"OpenAI structured generation failed: {e}") from e

    def _build_messages(self, prompt: str, context: Optional[str] = None) -> list[dict[str, str]]:
        """Constructs the message list for chat completions."""
        messages = []

        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _post_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Helper to post to chat/completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())
