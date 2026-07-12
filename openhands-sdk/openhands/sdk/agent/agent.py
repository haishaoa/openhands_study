from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import PrivateAttr, ValidationError, model_validator

import openhands.sdk.security.analyzer as analyzer
import openhands.sdk.security.risk as risk
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.agent.critic_mixin import CriticMixin
from openhands.sdk.agent.parallel_executor import ParallelToolExecutor
from openhands.sdk.agent.response_dispatch import (
    LLMResponseType,
    ResponseDispatchMixin,
    classify_response,
)
from openhands.sdk.agent.utils import (
    amake_llm_completion,
    aprepare_llm_messages,
    fix_malformed_tool_arguments,
    make_llm_completion,
    normalize_tool_call,
    parse_tool_call_arguments,
    prepare_llm_messages,
)
from openhands.sdk.context.prompts.presets import create_registry
from openhands.sdk.conversation import (
    CancellationToken,
    ConversationCallbackType,
    ConversationState,
    ConversationTokenCallbackType,
    LocalConversation,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    Event,
    MessageEvent,
    ObservationEvent,
    SystemPromptEvent,
    TokenEvent,
    UserRejectObservation,
)
from openhands.sdk.event.condenser import (
    Condensation,
    CondensationRequest,
)
from openhands.sdk.llm import (
    LLMResponse,
    Message,
    MessageToolCall,
    ReasoningItemModel,
    RedactedThinkingBlock,
    TextContent,
    ThinkingBlock,
)
from openhands.sdk.llm.exceptions import (
    FunctionCallValidationError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.observability.laminar import (
    maybe_init_laminar,
    observe,
    should_enable_observability,
)
from openhands.sdk.observability.utils import extract_action_name
from openhands.sdk.tool import (
    Action,
    Observation,
)


if TYPE_CHECKING:
    from openhands.sdk.tool import ToolDefinition
from openhands.sdk.mcp.tool import MCPToolDefinition
from openhands.sdk.tool.builtins import (
    FinishAction,
    FinishTool,
    ThinkAction,
)

logger = get_logger(__name__)


class Agent(CriticMixin, ResponseDispatchMixin, AgentBase):
    """Main agent implementation for OpenHands.

    The Agent class provides the core functionality for running AI agents that can
    interact with tools, process messages, and execute actions. It inherits from
    AgentBase and implements the agent execution logic. Critic-related functionality
    is provided by CriticMixin.

    Attributes:
        llm: The language model instance used for reasoning.
        tools: List of tools available to the agent.
        system_prompt: Inline system prompt string. When provided the agent
            uses this text verbatim instead of rendering from a template.
            Mutually exclusive with a non-default ``system_prompt_filename``.
            **Not recommended** unless you know what you are doing (e.g.
            customising agent behaviour for a completely different task) —
            this will override OpenHands' built-in system instructions.
        system_prompt_filename: Jinja2 template filename resolved relative to
            the agent's prompts directory, or an absolute path. Defaults to
            ``"system_prompt.j2"``.
        system_prompt_kwargs: Extra kwargs forwarded to the Jinja2 template.

    Example:
        ```python
        from openhands.sdk import LLM, Agent, Tool
        from pydantic import SecretStr

        llm = LLM(model="gpt-5.5", api_key=SecretStr("key"))
        tools = [Tool(name="TerminalTool"), Tool(name="FileEditorTool")]
        agent = Agent(llm=llm, tools=tools)
        ```

        To override the system prompt entirely::

            agent = Agent(
                llm=llm,
                tools=tools,
                system_prompt="You are a helpful coding assistant.",
            )
    """
    _parallel_executor: ParallelToolExecutor = PrivateAttr(
        default_factory=ParallelToolExecutor
    )

    @observe(name="agent.step", ignore_inputs=["state", "on_event"])
    def step(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        state = conversation.state
        # 检查待处理的操作（隐式确认）
        # 并在采样新操作之前执行这些操作。
        pending_actions = ConversationState.get_unmatched_actions(state.events)
        if pending_actions:
            logger.info(
                "Confirmation mode:Executing %d pending action(s)",
                len(pending_actions),
            )
            self._execute_actions(conversation, pending_actions, on_event)
            return

        # 检查最后一条用户消息是否被 UserPromptSubmit 钩子阻止
        # 如果是，则跳过处理并将对话标记为已结束
        if state.last_user_message_id is not None:
            reason = state.pop_blocked_message(state.last_user_message_id)
            if reason is not None:
                logger.info(f"User message blocked by hook:{reason}")
                state.execution_status = ConversationExecutionStatus.FINISHED
                return
        elif state.blocked_messages:
            logger.info(
                "已阻止的消息存在，但 last_user_message_id 为 None;"
                "跳过对旧版对话状态的钩子检查。"
            )

        # 从缓存的、增量维护的视图中准备 LLM 消息。
        # 参见 https://github.com/OpenHands/software-agent-sdk/issues/3053。
        _messages_or_condensation = prepare_llm_messages(
            state.view, condenser=self.condenser, llm=self.llm
        )

        # 在代理采样另一个动作之前，处理压缩事件
        if isinstance(_messages_or_condensation, Condensation):
            on_event(_messages_or_condensation)
            return

        _messages = _messages_or_condensation

        logger.debug(
            "Sending messages to LLM:"
            f"{json.dumps([m.model_dump() for m in _messages[1:]], indent=2)}"
        )

        try:
            llm_response = make_llm_completion(
                self.llm,
                _messages,
                tools=list(self.tools_map.values()),
                on_token=on_token
            )
        except FunctionCallValidationError as e:
            logger.warning(f"LLM generated malformed function call:{e}")
            error_message = MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=str(e))]
                ),
            )
            on_event(error_message)
            return
        except LLMContentPolicyViolationError as e:
            # Content-policy blocks are deterministic; nudge the model and let the
            # run loop continue instead of emitting a fatal error.
            logger.warning(f"LLM output blocked by content filter: {e}")
            on_event(
                MessageEvent(
                    source="user",
                    llm_message=Message(
                        role="user",
                        content=[
                            TextContent(
                                text=(
                                    "Your previous response was blocked by the "
                                    "model's content filter. Please continue, "
                                    "rephrasing to avoid the flagged content."
                                )
                            )
                        ],
                    ),
                )
            )
            return
        except LLMMalformedConversationHistoryError as e:
            # The provider rejected the current message history as structurally
            # invalid (for example, broken tool_use/tool_result pairing). Route
            # this into condensation recovery, but keep the logs distinct from
            # true context-window exhaustion so upstream event-stream bugs remain
            # visible.
            if (
                self.condenser is not None
                and self.condenser.handles_condensation_requests()
            ):
                logger.warning(
                    "LLM raised malformed conversation history error, "
                    "triggering condensation retry with condensed history: "
                    f"{e}"
                )
                # The incremental view may itself be the source of the
                # malformed history.  Re-derive with full enforcement so
                # the condenser operates on a clean view.
                state.rebuild_view()
                on_event(CondensationRequest())
                return
            logger.warning(
                "LLM raised malformed conversation history error but no "
                "condenser can handle condensation requests. This usually "
                "indicates an upstream event-stream or resume bug: "
                f"{e}"
            )
            raise e
        except LLMContextWindowExceedError as e:
            # If condenser is available and handles requests, trigger condensation
            if (
                self.condenser is not None
                and self.condenser.handles_condensation_requests()
            ):
                logger.warning(
                    "LLM raised context window exceeded error, triggering condensation"
                )
                on_event(CondensationRequest())
                return
            # No condenser available or doesn't handle requests; log helpful warning
            self._log_context_window_exceeded_warning()
            raise e

        # LLMResponse already contains the converted message and metrics snapshot
        message: Message = llm_response.message
        response_type = classify_response(message)

        match response_type:
            case LLMResponseType.TOOL_CALLS:
                self._handle_tool_calls(
                    message, llm_response, conversation, state, on_event
                )
            case LLMResponseType.CONTENT:
                self._handle_content_response(
                    message, llm_response, conversation, state, on_event
                )
            case LLMResponseType.REASONING_ONLY | LLMResponseType.EMPTY:
                self._handle_no_content_response(
                    message, llm_response, conversation, state,
                    on_event, response_type=response_type
                )

    @observe(name="agent.astep", ignore_input=["state", "on_event"])
    async def astep(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        """Async variant of :meth:`step`.

        The LLM completion is performed asynchronously via
        :func:`amake_llm_completion`.  Tool dispatch uses
        :meth:`_aexecute_actions` which runs each tool call in its own
        thread via :func:`asyncio.loop.run_in_executor` and schedules
        parallel calls with :func:`asyncio.gather`, keeping the event
        loop responsive during blocking tool I/O.
        """
        state = conversation.state
        # Check for pending actions (implicit confirmation)
        pending_actions = ConversationState.get_unmatched_actions(state.events)
        if pending_actions:
            logger.info(
                "Confirmation mode: Executing %d pending action(s)",
                len(pending_actions),
            )
            await self._aexecute_actions(conversation, pending_actions, on_event)
            return

        if state.last_user_message_id is not None:
            reason = state.pop_blocked_message(state.last_user_message_id)
            if reason is not None:
                logger.info(f"User message blocked by hook: {reason}")
                state.execution_status = ConversationExecutionStatus.FINISHED
                return

        elif state.blocked_messages:
            logger.debug(
                "Blocked messages exist but last_user_message_id is None; "
                "skipping hook check for legacy conversation state."
            )

        # Prepare LLM messages from the cached, incrementally-maintained view.
        # See https://github.com/OpenHands/software-agent-sdk/issues/3053.
        _messages_or_condensation = await aprepare_llm_messages(
            state.view, condenser=self.condenser, llm=self.llm
        )

        if isinstance(_messages_or_condensation, Condensation):
            on_event(_messages_or_condensation)
            return

        _messages = _messages_or_condensation

        logger.debug(
            "Sending messages to LLM: "
            f"{json.dumps([m.model_dump() for m in _messages[1:]], indent=2)}"
        )

        try:
            llm_response = await amake_llm_completion(
                self.llm,
                _messages,
                tools=list(self.tools_map.values()),
                on_token=on_token,
            )
        except FunctionCallValidationError as e:
            logger.warning(f"LLM generated malformed function call: {e}")
            error_message = MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=str(e))],
                ),
            )
            on_event(error_message)
            return
        except LLMContentPolicyViolationError as e:
            # Content-policy blocks are deterministic; nudge the model and let the
            # run loop continue instead of emitting a fatal error.
            logger.warning(f"LLM output blocked by content filter: {e}")
            on_event(
                MessageEvent(
                    source="user",
                    llm_message=Message(
                        role="user",
                        content=[
                            TextContent(
                                text=(
                                    "Your previous response was blocked by the "
                                    "model's content filter. Please continue, "
                                    "rephrasing to avoid the flagged content."
                                )
                            )
                        ],
                    ),
                )
            )
            return
        except LLMMalformedConversationHistoryError as e:
            # The provider rejected the current message history as
            # structurally invalid (for example, broken
            # tool_use/tool_result pairing).  Route this into
            # condensation recovery, but keep the logs distinct from
            # true context-window exhaustion so upstream event-stream
            # bugs remain visible.
            if (
                self.condenser is not None
                and self.condenser.handles_condensation_requests()
            ):
                logger.warning(
                    "LLM raised malformed conversation history error, "
                    "triggering condensation retry with condensed "
                    "history: %s",
                    e,
                )
                # Mirror step(): re-derive the cached view with full
                # enforcement before the condensation retry.
                state.rebuild_view()
                on_event(CondensationRequest())
                return
            logger.warning(
                "LLM raised malformed conversation history error but "
                "no condenser can handle condensation requests. This "
                "usually indicates an upstream event-stream or resume "
                "bug: %s",
                e,
            )
            raise e
        except LLMContextWindowExceedError as e:
            # If condenser is available and handles requests, trigger
            # condensation
            if (
                self.condenser is not None
                and self.condenser.handles_condensation_requests()
            ):
                logger.warning(
                    "LLM raised context window exceeded error, triggering condensation"
                )
                on_event(CondensationRequest())
                return
            # No condenser available; log helpful warning
            self._log_context_window_exceeded_warning()
            raise e

        message: Message = llm_response.message
        response_type = classify_response(message)

        match response_type:
            case LLMResponseType.TOOL_CALLS:
                await self._ahandle_tool_calls(
                    message, llm_response, conversation, state, on_event
                )
            case LLMResponseType.CONTENT:
                self._handle_content_response(
                    message, llm_response, conversation, state, on_event
                )
            case LLMResponseType.REASONING_ONLY | LLMResponseType.EMPTY:
                self._handle_no_content_response(
                    message,
                    llm_response,
                    conversation,
                    state,
                    on_event,
                    response_type=response_type,
                )
