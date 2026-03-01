from litellm import batch_completion
import os
import time
import pickle
from litellm.types.utils import ModelResponse, ChatCompletionTokenLogprob

from utils.config import LLMConfig
from utils.type import LMOutput

class LM:
    def __init__(
        self,
        config: LLMConfig
    ):
        self.config = config


    def __call__(
        self,
        prompts: list[list[dict]]
    ):
        start = time.perf_counter()
        if self.config.logprobs is not None:

            kwargs = {
                "logprobs": True,
                "top_logprobs": self.config.logprobs
            }
            responses = batch_completion(
                model = self.config.model_name,
                messages = prompts,
                api_base = self.config.base_url,
                api_key = self.config.api_key,
                max_tokens = self.config.max_tokens,
                temperature = self.config.temperature,
                stop = self.config.stop,
                **kwargs
            )
        else: 
            responses = batch_completion(
                model = self.config.model_name,
                messages = prompts,
                api_base = self.config.base_url,
                api_key = self.config.api_key,
                max_tokens = self.config.max_tokens,
                temperature = self.config.temperature,
                stop = self.config.stop,
            )
        end = time.perf_counter()

        if self.config.store:
            if not os.path.exists(self.config.output_path):
                os.makedirs(self.config.output_path)
    
            with open(os.path.join(self.config.output_path, self.config.output_name), 'w') as f:
                for i, response in enumerate(responses):
                    f.write(f"--- Response {i + 1} ---\n")
                    if isinstance(response, Exception):
                        f.write(f"[ERROR] {response.__class__.__name__}: {str(response)}\n")
                    elif isinstance(response, dict):
                        for key, value in response.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write(str(response) + "\n")
                    f.write("\n")
            with open(os.path.join(self.config.output_path, f"{self.config.output_name}_raw.pkl"), 'wb') as f:
                pickle.dump(responses, f)

        completions = []
        token_logprobs = []

        for r in responses:
            choice = r['choices'][0]['message']['content']
            completions.append(choice)
            if self.config.logprobs:
                logprobs = self.get_top_choice_logprobs(r)
                token_logprobs.append(logprobs)
        
        total_prompt_tokens = sum(r['usage']['prompt_tokens'] for r in responses)
        total_completion_tokens = sum(r['usage']['completion_tokens'] for r in responses)
        total_tokens = total_prompt_tokens + total_completion_tokens

        usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        }

        if self.config.store:
            with open(os.path.join(self.config.output_path, f"{self.config.output_name}_completions.pkl"), 'wb') as f:
                pickle.dump(completions, f)

        result = LMOutput(
            completions = completions,
            token_logprobs = token_logprobs if self.config.logprobs is not None else None,
            usage = usage,
            time_usage = end - start
        )

        return result

    def get_top_choice_logprobs(
        self,
        response: ModelResponse
    ) -> list[ChatCompletionTokenLogprob]:
        top_choice = response.choices[0]
        return top_choice.logprobs.content
