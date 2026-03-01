import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import gc
from tqdm import tqdm
import torch.nn.functional as F
from utils.config import RMConfig

class RM():
    
    def __init__(
        self,
        config: RMConfig
    ) -> None:
        self.embedding_model = config.embedding_model
        self.max_batch_size = config.max_batch_size
        self.normalize_embeddings = config.normalize_embeddings
        self.device = config.device
        self.max_length = config.max_length
    
    def __call__(
        self,
        texts: list[str]
    )-> np.ndarray:
        return self.get_embedding(texts)

    def get_embedding(
        self,
        docs: list[str],
    ) -> np.ndarray:
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        model = AutoModel.from_pretrained(self.embedding_model).to(self.device)
        model.eval()

        all_embeddings = []
        gc.collect()
        torch.cuda.empty_cache()

        max_length = self.max_length

        for doc in tqdm(docs, desc = "get embedding"):
            tokens = tokenizer(
                doc,
                return_tensors = 'pt',
                truncation = False
            )
            # total length
            input_ids = tokens['input_ids'][0]
            attention_mask = tokens['attention_mask'][0]

            chunk_embeddings = []
            if len(input_ids) >= max_length:
                print(f"Warning: the length of the document exceeds the max length {max_length}. It will be chunked.")
            for i in range(0, len(input_ids), max_length):
                chunk_ids = input_ids[i: i + max_length]
                chunk_mask = attention_mask[i: i + max_length]

                batch = {
                    'input_ids': chunk_ids.unsqueeze(0).to(self.device),
                    'attention_mask': chunk_mask.unsqueeze(0).to(self.device)
                }

                with torch.inference_mode():
                    outputs = model(**batch)
                    last_hidden_state = outputs.last_hidden_state
                    mask = batch['attention_mask']

                    masked = last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
                    embedding = masked.sum(dim = 1) / mask.sum(dim=1)[..., None]
                    embedding = F.normalize(embedding, p = 2, dim = 1)
                    chunk_embeddings.append(embedding.cpu())
            doc_embedding = torch.stack(chunk_embeddings, dim = 0).mean(dim = 0)
            doc_embedding = F.normalize(doc_embedding, p = 2, dim = 1)
            all_embeddings.append(doc_embedding)
        return torch.cat(all_embeddings, dim = 0).numpy()
                