import pandas as pd

from utils.settings import settings
from utils.config import JoinConfig
from utils.config import JoinType
from utils.preprocess import process_user_instruction_join, generate_prompt_pair_join
from utils.postprocess import filter_postprocess
from operators.join.filter_join import unnested_join, nested_join



def product_join(
    result_df: pd.DataFrame,
    config: JoinConfig
):
    print(f'Basic join execution starts, totally {len(result_df)} LLM calls')

    if 'prompt' not in result_df.columns:
        raise AttributeError('Prompt does not exist in result_df')

    prompts = result_df['prompt'].tolist()
    lm = settings.lm
    response = lm(prompts)
    results = filter_postprocess(response.completions)
    
    results.attrs['LLM_execute_time'] = response.time_usage

    if config.return_all:
        result_df['joined_result'] = results
        return result_df
    else:
        mask = [res == '1' for res in results]
        return result_df[mask]


def cartesian_product(
    left: pd.DataFrame,
    right: pd.DataFrame,
    instruction: str,
    left_col: str,
    right_col: str
):
    # do a cartesian_product between left and right, and generate a new column prompt by apply generate_prompt function in each row of column (left_col, right_col)
    left['_tmp_key'] = 1
    right['_tmp_key'] = 1

    merged = pd.merge(left, right, on = '_tmp_key').drop(columns=['_tmp_key'])

    def format_row(row, col):
        return f'{col}: {row[col]}'

    merged['prompt'] = merged.apply(
        lambda row: generate_prompt_pair_join(
            format_row(row, left_col),
            format_row(row, right_col),
            right_col,
            instruction
        ),
        axis = 1
    )

    return merged


def semantic_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    user_instruction: str,
    config: JoinConfig
) -> pd.DataFrame:
    cleaned_instruction, left_col, right_col = process_user_instruction_join(user_instruction)

    if left_col not in left.columns:
        raise ValueError(f'Not found {left_col} in left columns: {left.columns}')
    if right_col not in right.columns:
        raise ValueError(f'Not found {right_col} in right columns: {right.columns}')
    # should be a dataframe containing columns: l_id, r_id, list[dict] (prompts)

    if config.type == JoinType.NEST:
        return nested_join(
            left,
            right,
            cleaned_instruction,
            left_col,
            right_col,
            config
        )
    else:
        return unnested_join(
            left,
            right,
            cleaned_instruction,
            left_col,
            right_col,
            config
        )
    
