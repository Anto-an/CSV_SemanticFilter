import re
import pandas as pd
import sys
from collections import defaultdict
from utils.config import JoinConfig

def process_user_instruction(
    instruction: str
) -> tuple[str, list[str]]:
    pattern = r'\{(.*?)\}'

    matches = re.findall(pattern, instruction)
    cleaned = re.sub(pattern, r'\1', instruction)

    return cleaned, matches

# the instruction contain two {...} {...}, extra ... from {}. One should end with :left, the other should end with :right. Return the content before :left as left col. Same as :right
def process_user_instruction_join(
    instruction: str
) -> tuple[str, str, str]:
    pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, instruction)

    if len(matches) != 2:
        raise ValueError("Instruction must contain exactly two {...} placeholders.")
    
    left_col = None
    right_col = None

    for match in matches:
        if match.endswith(":left"):
            left_col = match[:-5]
        elif match.endswith(":right"):
            right_col = match[:-6]
        else:
            raise ValueError(f"Invalid placeholder format: {{{match}}}. Must end with ':left' or 'right'.")
    
    if not left_col or not right_col:
        raise ValueError("Both ':left' and ':right' placeholders must be provided")
    
    cleaned_instruction = instruction
    cleaned_instruction = cleaned_instruction.replace(f"{{{left_col}:left}}", f"{left_col}")
    cleaned_instruction = cleaned_instruction.replace(f"{{{right_col}:right}}", f"{right_col}")

    return cleaned_instruction, left_col, right_col
    

def df2text(
    data: pd.DataFrame,
    columns: list[str],
    include_colname: bool = True
) -> list[str]:
    if not columns:
        raise ValueError("No columns specified for df2text conversion.")
    

    # Check if all specified columns exist in the DataFrame
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    if include_colname:
        def format_row(row):
            return '\n'.join([f'{col}: {str(row[col])}' for col in columns])
    else:
        def format_row(row):
            return '\n'.join([str(row[col]) for col in columns])
    
    return data.apply(format_row, axis=1).tolist()

def generate_filter_prompts(
    data: pd.DataFrame,
    instruction: str,
    columns: list[str],
    include_colname: bool = True
) -> list[list[dict]]:
    text_data = df2text(data, columns, include_colname)

    system_message = {
        "role": "system",
        "content": (
            "You are an advanced AI trained to judge whether a provided claim is supported by the given context.\n"
            "Please follow these rules:\n"
            "1. If the context supports the claim, respond with '1'.\n"
            "2. If the context does not support the claim or is irrelevant, respond with '0'.\n"
            "3. Your response should be a single number, either '0' or '1'.\n"
            "4. Do not include any additional information or explanations."
        )
    }

    prompts = []
    for context in text_data:
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\nClaim: {instruction}\nAnswer:"
        }
        prompts.append([system_message, user_message])

    return prompts

def generate_topk_bestone_prompts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_col: str,
    right_col: str,
    instruction: str,
    tmp_indices: dict,
    config: JoinConfig
):
    print('Best one prompt')
    left_data = df2text(left, [left_col])
    right_data = df2text(right, [right_col])

    topk = config.topk

    task_description = (
        "You are an advanced AI assistant. "
        f"Your task is to select the single {right_col} that best satisfies the following instruction:\n" 
        f"Instruction: {instruction}\n\n"
    )

    # rules = (
    #     "Follow these rules strictly:\n"
    #     f"1. Select the single best {right_col} ID that most clearly follows the instruction: {instruction}.\n"
    #     f"2. If none of the {right_col} options clearly follow the instruction, return only '0'.\n"
    #     f"3. Your response must contain only the selected ID or '0' — no explanations or extra text.\n"
    #     f"4. If a/an {right_col} is vague, ambiguous, or partially relevant, do not select it.\n"
    # )

    rules = (
        "Follow these rules strictly:\n"
        f"1. Review the given {right_col} options carefully.\n"
        f"2. Select the ONE {right_col} ID that most clearly satisfies the instruction.\n"
        f"3. If none of the options clearly satisfy the instruction, output only '0'.\n"
        f"4. Do not select vague, ambiguous, or partially relevant options.\n"
        f"5. Your output must contain ONLY the chosen ID (e.g., '2') or '0'.\n"
        "6. Do not include explanations, reasoning, or any extra text.\n\n"
    )


    system_message = {
        "role": "system",
        "content": task_description + rules
    }

    prompts = []
    indices = []
    related_right_idx = []
    # print(f'Tmp indices: {str(tmp_indices)}')
    for left_idx, right_pairs in tmp_indices.items():
        if left_idx >= len(left_data):
            raise IndexError(f'Out of length of left_data.\nLeft data: {len(left_data)}\nl_idx: {left_idx}')
        left_content = left_data[left_idx]
        for batch_start in range(0, len(right_pairs), topk):
            batch_end = batch_start + topk
            batch_pairs = right_pairs[batch_start : batch_end]
            related_right_idx.append(batch_pairs)
            indices.append(left_idx)
            
            right_contents = []
            for r_idx, _ in batch_pairs:
                if r_idx >= len(right_data):
                    raise IndexError(f"Out of length of right data.\nRight data: {len(right_data)}\nr_idx: {r_idx}")
                right_contents.append(right_data[r_idx])
        
            user_contents = (
                f"Left item:\n{left_content}\n\n"
                "Candidate options: \n"
            )
            for i, content in enumerate(right_contents):
                user_contents += f"ID {i + 1}: {content}\n"
            user_contents += "Answer:"
        
            user_message = {
                "role": "user",
                "content": user_contents
            }
            prompt = [system_message, user_message]
            prompts.append(prompt)
    
    if config.store_prompts:
        import pickle
        with open(config.output_prompts_path, 'wb') as f:
            # for i, prompt in enumerate(prompts):
            #     f.write(f'-----Prompt {i}-------\n')
            #     f.write(str(prompt))
            pickle.dump(prompts, f)
            sys.exit(0)

    print(f'Length of prompts: {len(prompts)}')
    return prompts, indices, related_right_idx

    task_description = (
    "You are an advanced AI assistant.\n"
    "Your task is to evaluate multiple candidate items (Right) against a reference item (Left), "
    f"based on the following instruction:\n\nInstruction: {instruction}\n\n"
    )

    rules = (
        "Please follow these rules strictly:\n"
        f"1. Select the single best {right_col} ID that most clearly satisfies the instruction.\n"
        f"2. If none of the {right_col} options clearly satisfy the instruction, return only '0'.\n"
        f"3. Your response must contain only the selected ID or '0' — no explanations or extra text.\n"
        f"4. Do not select any {right_col} that is vague, ambiguous, or only partially relevant.\n"
    )

    user_contents = (
    f"Reference ({left_col}):\n{left_content}\n\n"
    f"Candidate {right_col} options:\n"
    )

    for i, content in enumerate(right_contents):
        user_contents += f"ID {i + 1}: {content}\n"

    user_contents += "\nPlease select the best matching ID according to the instruction above.\nAnswer:"


    system_message = {
        "role": "system",
        "content": task_description + rules
    }


def generate_prompt_pair_join(
    left: str,
    right: str,
    right_col: str,
    instruction: str,
    config: JoinConfig
)->list:
    task_description = (
        "You are an advanced AI assistant."
        f"Your task is to determine whether {instruction}\n"
    )

    rules = (
        "Please follow these rules: \n"
        f"1. If the {right_col} follow the instruction {instruction}, response with '1'.\n"
        f"2. If the {right_col} does not follow the instruction: {instruction}, response with '0'.\n"
        f"3. Your response should be a single number: '0' or '1'.\n"
        "4. Do no explain your answer or provide any additional text. \n"
    )

    system_message = {
        'role': 'system',
        'content': task_description + rules
    }

    user_message = {
        'role': 'user',
        'content': f'Left:\n{left}\nRight:\n{right}\nAnswer:'
    }

    prompt = [system_message, user_message]
    return prompt

def generate_topk_pair_prompts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_col: str,
    right_col: str,
    instruction: str,
    tmp_indices: dict,
    config: JoinConfig
):
    print('Pair prompt')
    left_data = df2text(left, [left_col])
    right_data = df2text(right, [right_col])

    task_description = (
        "You are an advanced AI assitant."
        f"Your task is to determine wheter {instruction}\n"
    )

    rules = (
        "Please follow these rules: \n"
        f"1. If the {right_col} follow the instruction: {instruction}, response with '1'.\n"
        f"2. If the {right_col} does not follow the instruction: {instruction}, response with '0'.\n"
        f"3. Your response should be a single number: '0' or '1'.\n"
        "4. Do no explain your answer or provide any addtional text.\n"
    )

    system_message = {
        "role": "system",
        "content": task_description + rules
    }

    prompts = []
    indices = []

    for left_idx, right_pairs in tmp_indices.items():
        if left_idx >= len(left_data):
            raise IndexError(f'Out of length of left_data.\nLeft data: {len(left_data)}\nl_idx: {left_idx}')
        
        left_content = left_data[left_idx]
        for r_idx, score in right_pairs:
            if r_idx >= len(right_data):
                raise IndexError(f'Out the length of right data.\nRight data: {len(right_data)}\nr_idx: {r_idx}')
            content = right_data[r_idx]
            user_message = {
                "role": "user",
                "content": f"Left:\n{left_content}\nRight:\n{content}\nAnswer:"
            }
            prompt = [system_message, user_message]
            prompts.append(prompt)
            indices.append((left_idx, r_idx, score))
    
    if config.store_prompts:
        with open(config.output_prompts_path, 'w') as f:
            for i, prompt in enumerate(prompts):
                f.write(f'-----Prompt {i}-------\n')
                f.write(str(prompt))
    print(f'Length of prompts: {len(prompts)}')
    
    return prompts, indices
