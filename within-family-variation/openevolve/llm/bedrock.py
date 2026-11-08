"""
Bedrock API interface for LLMs (DeepSeek-R1 via the Bedrock Converse API).

Matches the paper's stated methodology ("powered by DeepSeek-R1") rather than
the repo's shipped config.yaml (a 50/50 deepseek-chat/deepseek-reasoner
ensemble, which the paper's text does not mention -- see conversation notes).
Uses Bedrock's newer bearer-token auth (AWS_BEARER_TOKEN_BEDROCK env var),
not classic IAM/SigV4 credentials.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import boto3

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class BedrockLLM(LLMInterface):
    """LLM interface using AWS Bedrock's Converse API"""

    def __init__(self, model_cfg: Optional[dict] = None):
        self.model = model_cfg.name  # e.g. "us.deepseek.r1-v1:0"
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.region = getattr(model_cfg, "api_base", None) or "us-east-1"
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # boto3 auto-detects AWS_BEARER_TOKEN_BEDROCK from the environment --
        # no access key/secret needed with Bedrock's bearer-token auth.
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            logger.info(f"Initialized Bedrock LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        formatted_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]} for m in messages
        ]
        params = {
            "modelId": self.model,
            "messages": formatted_messages,
            "inferenceConfig": {
                "maxTokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "topP": kwargs.get("top_p", self.top_p),
            },
        }
        if system_message:
            params["system"] = [{"text": system_message}]

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(self._call_api(params), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.converse(**params)
        )
        content_blocks = response["output"]["message"]["content"]
        # DeepSeek-R1 on Bedrock returns a reasoningContent block plus a text
        # block -- we want only the final text answer, not the chain-of-thought.
        text_parts = [b["text"] for b in content_blocks if "text" in b]
        return "".join(text_parts)
