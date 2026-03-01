from datetime import datetime
from utils.type import VoteType, JoinType, EstimationType, KNNType
import os

LLM_LARGE_ID = "openai/RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16"
LLM_SMALL_ID = "openai/meta-llama/Llama-3.1-8B-Instruct"
LLM_HELP_ID = "openai/meta-llama/Llama-3.2-3B-Instruct"

class FilterConfig:
    def __init__(
        self,
        vote: VoteType = VoteType.NONE,
        vote_upper: float = 0.9,
        vote_lower: float = 0.1,
        cluster_num: int = 4,
        sample_ratio: float = 0.01,
        max_samples: int = 1000,
        min_samples: int = 100,
        batch_size: int = 64,
        return_all: bool = True,
        new_column_name: str = "filtered_output",
        store_prompts: bool = False,
        output_prompts_path: str | None = None,
        max_depth: int = 3,
        final_vote: bool = True,
        use_mean: bool = False
        
    ):
        self.vote = vote
        self.vote_upper = vote_upper
        self.vote_lower = vote_lower
        self.cluster_num = cluster_num
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.return_all = return_all
        self.new_column_name = new_column_name
        self.batch_size = batch_size
        self.store_prompts = store_prompts
        self.output_prompts_path = output_prompts_path if output_prompts_path is not None else f"data/filter_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.max_depth = max_depth
        self.final_vote = final_vote
        self.use_mean = use_mean

class JoinConfig:
    def __init__(
        self,
        type: JoinType = JoinType.PRODUCT,
        estimate_type: EstimationType = EstimationType.GRAPH,
        use_ann: bool = False,
        topk: int = 3,
        return_all: bool = True,
        tau: int = -1,
        max_iteration: int = 10,
        best_one_prompt: bool = True,
        store_prompts: bool = True,
        output_prompts_path: str | None = None,
        filter_config: FilterConfig | None = None,
    ):
        self.type = type
        self.estimate_type = estimate_type
        self.use_ann = use_ann
        self.topk = topk
        self.return_all = return_all
        self.tau = tau
        self.max_iteration = max_iteration
        self.best_one_prompt = best_one_prompt
        self.store_prompts = store_prompts
        self.output_prompts_path = output_prompts_path if output_prompts_path is not None else f"data/join_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.statistic = {
            "LLM_call": [],
            "LLM_execute_time": [],
            "prompt_tokens": [],
            "completion_tokens": [],
            "matrix_completion_time": [],
            "num_sampled": 0,
            "linear_process": 0,
            "ratio": [],
            "cluster": 0
        }
        self.filter_config = filter_config

class LLMConfig:
    def __init__(
        self,
        model_name: str,
        base_url: str = os.environ['OPENAI_API_BASE'],
        api_key: str = os.environ['OPENAI_API_KEY'],
        max_tokens: int = 32,
        temperature: float = 0.7,
        logprobs: int | None = None,
        batch_size: int = 64,
        
        stop=['\n'],
        store: bool = False,
        output_path: str = 'experiments/filter/llm_output',
        output_name: str | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logprobs = logprobs
        self.batch_size = batch_size
        self.model_name = model_name
        self.stop = stop
        self.store = store
        self.output_path = output_path
        self.output_name = output_name if output_name is not None else datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"

class RMConfig:
    def __init__(
        self,
        embedding_model: str = 'intfloat/e5-large-v2',
        max_batch_size: int = 128,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        device: str = 'cuda'
    ):
        self.embedding_model = embedding_model
        self.max_batch_size = max_batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.max_length = max_length
        