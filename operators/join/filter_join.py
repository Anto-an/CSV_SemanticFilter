import pandas as pd

from utils.config import JoinConfig, FilterConfig
from operators.filter.filter_func import basic_filter
from operators.filter.optimized_filter import optimized_filter
from utils.preprocess import generate_filter_prompts
from utils.type import VoteType

def cartesian_product(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_col: str,
    right_col: str
) -> pd.DataFrame:
    # generate a new dataframe containing columns: content, left_idx, right_idx
    left['_tmp_key'] = 1
    right['_tmp_key'] = 1
    left['left_idx'] = left.index
    right['right_idx'] = right.index
    merged = pd.merge(left, right, on = '_tmp_key').drop(columns=['_tmp_key'])

    merged['content'] = merged.apply(
        lambda row: f'{left_col}: {row[left_col]}\n{right_col}: {row[right_col]}',
        axis = 1
    )

    merged = merged[['content', 'left_idx', 'right_idx']]
    return merged


def unnested_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    instruction: str,
    left_col: str,
    right_col: str,
    config: JoinConfig
)-> pd.DataFrame:
    data = cartesian_product(
        left,
        right,
        left_col,
        right_col
    )

    prompts = generate_filter_prompts(
        data,
        instruction,
        ['content']
    )

    if config.store_prompts:
        with open(config.output_prompts_path, 'wb') as f:
            import pickle
            pickle.dump(prompts, f)

    
    if config.filter_config is None:
        raise ValueError("filter_config must be provided for filter join")
    
    data.attrs['embedding_path'] = left.attrs.get('filter_embedding_path', None)

    result_df = filter_join(
        data,
        prompts,
        config,
        instruction
    )
    
    return generate_join_result(
        result_df,
    )

def generate_join_result(
    result_df: pd.DataFrame
) -> pd.DataFrame:
    # only preserve rows with filter_result == '1'
    if 'filtered_output' not in result_df.columns:
        raise ValueError("filter_result column not found in result_df")
    print(f'Length of result_df before filtering: {len(result_df)}')
    print(f'Length of result_df after filtering: {len(result_df[result_df["filtered_output"] == 1])}')
    filtered = result_df[result_df['filtered_output'] == 1]
    filtered['similarity'] = 1.0
    filtered['iteration'] = 0

    return filtered[['left_idx', 'right_idx']]


def nested_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    instruction: str,
    left_col: str,
    right_col: str,
    config: JoinConfig
):
    result_pairs = []
    for left_idx, row in left.iterrows():
        tmp_right = right.copy()
        left_text = row[left_col]
        augmented_instruction = f'{instruction}\n{left_col.upper()}: {left_text}'

        prompts = generate_filter_prompts(
            right,
            augmented_instruction,
            [right_col],
        )

        if config.store_prompts:
            with open(config.output_prompts_path, 'wb') as f:
                import pickle
                pickle.dump(prompts, f)
        if config.filter_config is None:
            raise ValueError("filter_config must be provided for filter join")
        
        result_df = filter_join(
            tmp_right,
            prompts,
            config,
            augmented_instruction
        )

        for right_idx, right_row in result_df.iterrows():
            filtered_output = right_row.get('filtered_output', None)
            if filtered_output is None:
                raise ValueError(f'Cannot get filtered_output')
            if filtered_output == 1:
                result_pairs.append((left_idx, right_idx))
    joined_df = pd.DataFrame(result_pairs, columns = ['left_idx', 'right_idx']) 

    return joined_df


        
def filter_join(
    data: pd.DataFrame,
    prompts: list[str],
    config: JoinConfig,
    instruction: str
):
    if config.filter_config.vote == VoteType.NONE:
        result_df = basic_filter(
            data,
            prompts,
            config.filter_config
        )
        config.statistic['num_sampled'] = len(prompts)
        config.statistic['linear_process'] = len(prompts)
    elif config.filter_config.vote in [VoteType.SIMVOTE, VoteType.EQUALVOTE]:
        
        
        result_df = optimized_filter(
            data,
            'content',
            prompts,
            config.filter_config,
            instruction
        )
        config.statistic['num_sampled'] += result_df.attrs.get('num_sampled', 0)
        config.statistic['linear_process'] += result_df.attrs.get('Linear_process', 0)
        config.statistic['cluster'] += result_df.attrs.get('cluster', 0)
        config.statistic['ratio'] += result_df.attrs.get('ratio', [])
    
    config.statistic['prompt_tokens'] += data.attrs.get('prompt_tokens', 0)
    config.statistic['completion_tokens'] += data.attrs.get('completion_tokens', 0)
    config.statistic['LLM_execute_time'] += data.attrs.get('LLM_execute_time', 0.0)

    return result_df