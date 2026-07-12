from __future__ import annotations

import copy
import importlib
import json
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from openhands.sdk.llm.fallback_strategy import FallbackStrategy
from openhands.sdk.llm.utils.model_info import get_litellm_model_info
from openhands.sdk.settings.metadata import SettingProminence, field_meta
from openhands.sdk.utils.pydantic_secrets import serialize_secret, validate_secret


if TYPE_CHECKING:  # type hints only, avoid runtime import cycle
    from openhands.sdk.llm.auth import SupportedVendor
    from openhands.sdk.llm.auth.openai import OpenAIAuthMethod
    from openhands.sdk.tool.tool import ToolDefinition

from openhands.sdk.llm.auth.openai import transform_for_subscription


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm

from typing import Final, cast

from litellm import (
    ChatCompletionToolParam,
    CustomStreamWrapper,
    ResponseInputParam,
    acompletion as litellm_acompletion,
    completion as litellm_completion,
)
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout as LiteLLMTimeout,
)
from litellm.responses.main import (
    aresponses as litellm_aresponses,
    responses as litellm_responses,
)
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ReasoningSummaryTextDeltaEvent,
    RefusalDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.utils import (
    create_pretrained_tokenizer,
    supports_vision,
    token_counter,
)

from openhands.sdk.llm.exceptions import (
    LLMContextWindowTooSmallError,
    LLMNoResponseError,
    is_prompt_cache_too_small,
    map_provider_exception,
)

# OpenHands utilities
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.message import (
    Message,
)
from openhands.sdk.llm.mixins.non_native_fc import NonNativeToolCallingMixin
from openhands.sdk.llm.options.chat_options import select_chat_options
from openhands.sdk.llm.options.responses_options import select_responses_options
from openhands.sdk.llm.streaming import (
    AnyTokenCallbackType,
    TokenCallbackType,
    _invoke_token_callback,
)
from openhands.sdk.llm.utils.image_inline import (
    amaybe_inline_image_urls,
    maybe_inline_image_urls,
)
from openhands.sdk.llm.utils.image_resize import maybe_resize_messages_for_provider
from openhands.sdk.llm.utils.litellm_provider import infer_litellm_provider
from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.llm.utils.model_features import get_features
from openhands.sdk.llm.utils.openhands_provider import (
    LiteLLMCallKwargs,
    canonicalize_openhands_llm_payload,
    litellm_call_kwargs,
)
from openhands.sdk.llm.utils.retry_mixin import RetryMixin
from openhands.sdk.llm.utils.telemetry import Telemetry
from openhands.sdk.llm.utils.vertex_preflight import assert_vertex_sdk_available
from openhands.sdk.logger import ENV_LOG_DIR, get_logger

logger = get_logger(__name__)

__all__ = ["LLM"]

# Minimum context window size required for OpenHands to function properly.
# Based on typical usage: system prompt (~2k) + conversation history (~4k)
# + tool definitions (~2k) + working memory (~8k) = ~16k minimum.
MIN_CONTEXT_WINDOW_TOKENS: Final[int] = 16384

# Environment variable to override the minimum context window check
ENV_ALLOW_SHORT_CONTEXT_WINDOWS: Final[str] = "ALLOW_SHORT_CONTEXT_WINDOWS"

# Default max output tokens when model info only provides 'max_tokens' (ambiguous).
# Some providers use 'max_tokens' for the total context window, not output limit.
# This cap prevents requesting output that exceeds the context window.
# 16384 is a safe default that works for most models (GPT-4o: 16k, Claude: 8k).
DEFAULT_MAX_OUTPUT_TOKENS_CAP: Final[int] = 16384

# Some providers (notably AWS Bedrock for Anthropic models) enforce
# ``input_tokens + max_tokens <= context_window`` on a single shared window.
# For these providers we clamp the per-request output budget at request time so
# our injected default ``max_tokens`` (which defaults to the model's full
# ``max_output_tokens`` -- e.g. 64k for Sonnet 4.5) cannot push large-input
# requests past the context window. See BerriAI/litellm#17900 for the upstream
# default change that made this manifest.
JOINT_BUDGET_SAFETY_MARGIN_TOKENS: Final[int] = 256
JOINT_BUDGET_MIN_OUTPUT_TOKENS: Final[int] = 1024

# Provider name prefixes known to enforce a joint input/output token budget.
# Kept as a narrow allowlist so direct providers (Anthropic, OpenAI, etc.) --
# which have independent input/output windows -- are not affected. We match by
# prefix because LiteLLM uses both ``bedrock`` (raw provider inference) and
# ``bedrock_converse`` (the Anthropic-on-Bedrock route, surfaced via the model
# registry) for the same underlying API.
_JOINT_BUDGET_PROVIDER_PREFIXES: Final[tuple[str, ...]] = ("bedrock",)

# Secret-bearing fields on LLM. Kept as a single source of truth so callers that
# need to walk secrets (e.g. cipher-aware decryption on the save path) stay in
# sync with the serializer below.
LLM_SECRET_FIELDS: Final[tuple[str, ...]] = (
    "api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
)

LLM_PROFILE_SCHEMA_VERSION: Final[int] = 1