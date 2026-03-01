from enum import Enum
from dataclasses import dataclass
from litellm.types.utils import ChatCompletionTokenLogprob

class VoteType(Enum):
    NONE = 'none'
    EQUALVOTE = 'equalvote'
    SIMVOTE = 'simvote'

class JoinType(Enum):
    NEST = 'product'
    UNNEST = 'filter'

@dataclass
class LMOutput:
    completions: list[str]
    token_logprobs: list[list[ChatCompletionTokenLogprob]] | None
    usage: dict
    time_usage: float