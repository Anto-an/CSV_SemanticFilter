import pandas as pd
from utils.config import FilterConfig
from utils.type import VoteType
from utils.preprocess import process_user_instruction, generate_filter_prompts
from utils.postprocess import filter_postprocess
from utils.settings import settings
from operators.filter.optimized_filter import optimized_filter

import pickle

def basic_filter(
    data: pd.DataFrame,
    prompts: list[list[dict]],
    config: FilterConfig,
) -> pd.DataFrame:
    print('Basic filter execution started.')
    print(f'Length of prompts: {len(prompts)}')
    lm = settings.lm
    # if lm.config.model_name == 'gpt':
    #     responses = 
    responses = lm(prompts)

    results = filter_postprocess(responses.completions)

    data.attrs['LLM_execute_time'] = responses.time_usage
    data.attrs['prompt_tokens'] = responses.usage['prompt_tokens']
    data.attrs['completion_tokens'] = responses.usage['completion_tokens']
    data.attrs['num_sampled'] = 0

    if config.return_all:
        if config.new_column_name is not None:
            if config.new_column_name in data.columns:
                raise ValueError(f"Column '{config.new_column_name}' already exists in the DataFrame.")
            data[config.new_column_name] = results
        else:
            if 'filtered_output' in data.columns:
                raise ValueError("Column 'filter_result' already exists in the DataFrame.")
            data['filter_result'] = results
        return data
    else:
        # Turn results into a boolean mask
        mask = [res == 1 for res in results]
        return data[mask]
    

def semantic_filter(
    data: pd.DataFrame,
    user_instruction: str,
    config: FilterConfig,
) -> pd.DataFrame:
    cleaned_instruction, cols = process_user_instruction(user_instruction)

    prompts = generate_filter_prompts(
        data,
        cleaned_instruction,
        cols
    )

    if config.store_prompts:
        with open(config.output_prompts_path, 'wb') as f:
            pickle.dump(prompts, f)
        return pd.DataFrame(columns=data.columns)

    if config.vote == VoteType.NONE:

        result = basic_filter(
            data,
            prompts,
            config
        )
        return result
    if config.vote in [VoteType.EQUALVOTE, VoteType.SIMVOTE]:
        return optimized_filter(
            data,
            cols[0],
            prompts,
            config,
            cleaned_instruction
        )


