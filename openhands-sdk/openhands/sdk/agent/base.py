from openhands.sdk.utils.models import DiscriminatedUnionMixin
from abc import ABC
from pydantic import ConfigDict, Field
from openhands.sdk.llm import LLM


class AgentBase(DiscriminatedUnionMixin, ABC):
    """
    OpenHands 代理的抽象基类。
    代理是无状态的，其功能应完全由其配置定义。
    此基类提供所有代理实现必须遵循的通用接口和功能。
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    llm: LLM = Field()
