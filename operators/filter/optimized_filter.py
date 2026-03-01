import os 
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from utils.config import FilterConfig
from utils.settings import settings
from utils.type import VoteType
from utils.postprocess import filter_postprocess

def sample(
    label_to_indices: dict[int, list[int]],
    sample_ratio: float,
    max_samples: int,
    min_samples: int
):
    # in each cluster, sample at least min_samples, and according to sample_ratio
    sampled_indices = {}
    for label, indices in label_to_indices.items():
        n_samples = int(max(len(indices) * sample_ratio, min_samples))
        if n_samples > len(indices):
            n_samples = len(indices)
        sampled_indices[label] = np.random.choice(indices, n_samples, replace=False).tolist()
    
    return sampled_indices

def get_embedding(
    data: pd.DataFrame,
    column_name: str,
):
    if not hasattr(data, 'attrs') or 'embedding_path' not in data.attrs:
        raise ValueError("DataFrame must have 'embedding_path' in its attrs for optimized_filter.")

    embedding_path = data.attrs['embedding_path']
    instruction_path = data.attrs.get('query_path', None)
   
    # print(f'Embedding path: {embedding_path}')
    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        rm = settings.rm
        embedding_data = data[column_name].astype(str).tolist()
        # print(f'Generating embeddings for column: {column_name}')
        embeddings = rm(embedding_data)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        with open(embedding_path, 'wb') as f:
            pickle.dump(embeddings, f)

    return embeddings

def get_llm_output(prompts: list[list[dict]]):
    lm = settings.lm
    llm_output = lm(prompts)
    result = filter_postprocess(llm_output.completions)
    return llm_output, result

def sample_and_process(
    data: pd.DataFrame,
    prompts: list[list[dict]],
    final_results: list[int],
    embeddings: np.ndarray,
    unsampled_indices: list[int],
    config: FilterConfig,
    flag
):
    if len(unsampled_indices) == 0:
        return None, None, None
    data.attrs['cluster'] += 1
    if len(unsampled_indices) <= config.min_samples:
        linear_process(data, prompts, unsampled_indices, final_results)
        return None, None, None

    if flag:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings[unsampled_indices])
    else:
        kmeans = KMeans(n_clusters=config.cluster_num, random_state=0).fit(embeddings[unsampled_indices])
    labels = kmeans.labels_

    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(unsampled_indices[idx])
    sampled_indices = sample(
        label_to_indices,
        sample_ratio = config.sample_ratio,
        max_samples = config.max_samples,
        min_samples = config.min_samples
    )
    sampled_prompts = {}
    sampled_results = {}
    all_sampled_prompts = []
    all_sampled_results = []
   
    
    for label, indices in sampled_indices.items():
        sampled_prompts[label] = [prompts[i] for i in indices]
        all_sampled_prompts.extend(sampled_prompts[label])
        
    llm_output, all_sampled_results = get_llm_output(all_sampled_prompts)
    data.attrs['LLM_execute_time'] += llm_output.time_usage
    data.attrs['prompt_tokens'] += llm_output.usage['prompt_tokens']
    data.attrs['completion_tokens'] += llm_output.usage['completion_tokens']
    data.attrs['num_sampled'] += len(all_sampled_prompts)
    # assign results back to each cluster
    current_index = 0
    for label, indices in sampled_indices.items():
        n = len(indices)
        sampled_results[label] = all_sampled_results[current_index:current_index + n]
        current_index += n

        for i, idx in enumerate(indices):
            # print(f'Cluster {label}: assigning sampled result {sampled_results[label][i]} to index {idx}')
            final_results[idx] = sampled_results[label][i]

    return label_to_indices, sampled_indices, sampled_results
    
def linear_process(
    data: pd.DataFrame,
    prompts: list[list[dict]],
    unsampled_indices: list[int],
    final_results: list[int],
):
    target_prompts = [prompts[i] for i in unsampled_indices]
    print(f'Length of linear process: {len(unsampled_indices)}')
    llm_output, result = get_llm_output(target_prompts)
    for i, idx in enumerate(unsampled_indices):
        final_results[idx] = result[i]
    data.attrs['LLM_execute_time'] += llm_output.time_usage
    data.attrs['prompt_tokens'] += llm_output.usage['prompt_tokens']
    data.attrs['completion_tokens'] += llm_output.usage['completion_tokens']
    data.attrs['Linear_process'] += len(unsampled_indices)
    data.attrs['num_sampled'] += len(unsampled_indices)
    
    return

def equal_vote(
    data: pd.DataFrame,
    prompts: list[list[dict]],
    embeddings: np.ndarray,
    unsampled_indices: list[int],
    config: FilterConfig,
    final_results: list[int],
    depth,
    flag = True,
): 
    if depth > config.max_depth:
        return
    print(f'Vote lower: {config.vote_lower}')
   
    label_to_indices, sampled_indices, sampled_results = sample_and_process(
        data,
        prompts,
        final_results,
        embeddings,
        unsampled_indices,
        config,
        flag
    )
    if label_to_indices is None:
        return 
    
    for label, indices in label_to_indices.items():
        sampled_results_in_cluster = sampled_results.get(label, [])
        sampled_indices_in_cluster = sampled_indices.get(label, [])
        count = sampled_results_in_cluster.count(1)
        # print(f'Cluster {label}: {count} positive out of {len(sampled_results_in_cluster)} sampled')
        total = len(sampled_results_in_cluster)
        if total == 0:
            raise ValueError("No sampled results in cluster, should not happen.")
        ratio = count / total
        data.attrs['ratio'].append((ratio, len(indices)))
        result = -1
        threshold = config.vote_lower

        if ratio >= 1 - threshold:
            result = 1
        elif ratio <= threshold:
            result = 0
        else:
            unsampled_sub_indices = list(set(indices) - set(sampled_indices_in_cluster))
            if depth + 1 >= config.max_depth:
                linear_process(
                    data,
                    prompts,
                    unsampled_sub_indices,
                    final_results
                )
            else:
                equal_vote(
                    data, 
                    prompts, 
                    embeddings, 
                    unsampled_sub_indices, 
                    config, 
                    final_results, 
                    depth + 1)
            continue
        for idx in indices:
            if idx not in sampled_indices_in_cluster:
                # print(f'Unsampled data: assigning result {result} to index {idx}')
                final_results[idx] = result
    return 


def sim_vote(
    data: pd.DataFrame,
    prompts: list[list[dict]],
    embeddings: np.ndarray,
    unsampled_indices: list[int],
    config: FilterConfig,
    final_results: list[int],
    depth: int,
    flag = True
):
    if depth > config.max_depth:
        return
    label_to_indices, sampled_indices, sampled_results = sample_and_process(
        data,
        prompts,
        final_results,
        embeddings,
        unsampled_indices,
        config,
        flag
    )
    if label_to_indices is None:
        return
    print(f'Length of label_to_indices: {len(label_to_indices)}')
    for label, indices in label_to_indices.items():
        # For unsampled data points in this cluster, calculate their similarity to the sampled points, and assign the label based on weighted vote
        unassigned_indices = []
        if label not in sampled_indices or len(sampled_indices[label]) == 0:
            raise ValueError("No sampled points in cluster, should not happen.")
        sampled_indices_in_cluster = sampled_indices[label]
        if label not in sampled_results or len(sampled_results[label]) == 0:
            raise ValueError("No sampled results in cluster, should not happen.")
        sampled_results_in_cluster = sampled_results[label]
        sampled_embeddings = embeddings[sampled_indices_in_cluster]
        unsampled_indices = [idx for idx in indices if idx not in sampled_indices_in_cluster]
        unsampled_embeddings = embeddings[unsampled_indices]

        if len(unsampled_indices) == 0:
            continue

        similarity_matrix = cosine_similarity(unsampled_embeddings, sampled_embeddings)
        similarity_matrix = (similarity_matrix + 1) / 2
        weights_matrix = similarity_matrix / np.sum(similarity_matrix, axis = 1, keepdims=True)
        votes = np.dot(weights_matrix, np.array(sampled_results_in_cluster))

        for i, idx in enumerate(tqdm(unsampled_indices, desc="Simvote")):
            vote = votes[i]
            if vote >= config.vote_upper:
                final_results[idx] = 1
            elif vote <= config.vote_lower:
                final_results[idx] = 0
            else:
                unassigned_indices.append(idx)
            
        if len(unassigned_indices) != 0:
            if depth + 1 >= config.max_depth:
                linear_process(
                    data,
                    prompts,
                    unassigned_indices,
                    final_results
                )
            else:
                sim_vote(
                    data, 
                    prompts, 
                    embeddings, 
                    unassigned_indices, 
                    config, 
                    final_results, 
                    depth + 1,
                )
    return 
    

def optimized_filter(
    data: pd.DataFrame,
    column_name: str,
    prompts: list[list[dict]],
    config: FilterConfig,
    instruction: str = None,
) -> pd.DataFrame:
    
    embeddings= get_embedding(data, column_name, instruction, config.use_mean)

    final_results = [-1] * len(data)

    initData(data)

    depth = 0
    if config.vote == VoteType.EQUALVOTE:
        equal_vote(
            data,
            prompts,
            embeddings,
            list(range(len(data))),
            config,
            final_results,
            depth = depth,
            flag = False,
        )
    elif config.vote == VoteType.SIMVOTE:
        sim_vote(
            data,
            prompts,
            embeddings,
            list(range(len(data))),
            config,
            final_results,
            depth = depth,
            flag = False
        )
    else:
        raise ValueError(f"Unknown vote type: {config.vote}")

    check_final_results(len(data), final_results)
    if config.new_column_name is not None:
        if config.new_column_name in data.columns:
            raise ValueError(f"Column '{config.new_column_name}' already exists in the DataFrame.")
        data[config.new_column_name] = final_results
    else:
        if 'filter_result' in data.columns:
            raise ValueError("Column 'filter_result' already exists in the DataFrame.")
        data['filter_result'] = final_results
    return data


def initData(data: pd.DataFrame):
    if not hasattr(data, 'attrs'):
        data.attrs = {}
    if 'LLM_execute_time' not in data.attrs:
        data.attrs['LLM_execute_time'] = 0.0
    if 'Linear_process' not in data.attrs:
        data.attrs['Linear_process'] = 0
    if 'num_sampled' not in data.attrs:
        data.attrs['num_sampled'] = 0
    if 'cluster' not in data.attrs:
        data.attrs['cluster'] = 0
    if 'ratio' not in data.attrs:
        data.attrs['ratio'] = []
    if 'prompt_tokens' not in data.attrs:
        data.attrs['prompt_tokens'] = 0
    if 'completion_tokens' not in data.attrs:
        data.attrs['completion_tokens'] = 0

def check_final_results(length: int, final_results: list[int]):
    if len(final_results) != length:
        raise ValueError("Final results length does not match data length.")
    for res in final_results:
        if res not in [0, 1]:
            print(f'Final results: {final_results}')
            raise ValueError("Final results contain invalid values.")