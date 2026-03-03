# Beyond Linear LLM Invocation: An Efficient and Effective Semantic Filter Paradigm

This repository contains the source code and technical report accompanying our **VLDB 2026** submission:

> **Beyond Linear LLM Invocation: An Efficient and Effective Semantic Filter Paradigm**

---

## Repository Structure

| Directory   | Description                          |
|-------------|--------------------------------------|
| `models/`   | Connection to LLM and embedding model|
| `operators/`| Operator implementations             |
| `utils/`    | Helper functions and utilities       |
| `data/`     | Benchmark datasets                   |
| `README.md` | Project documentation                |
| `TR.pdf`    | Technical Report                     |

## Set up

The following Python packages are required:
- litellm
- openai
- torch
- tqdm
- transformers
- sklearn

### Conda Environment Setup
We provide an `environment.yml' file for easy conda environment setup. Before creating the environment, updating the prefix in the last line to match your desired environment path.
```bash
conda env create -f environment.yml
```

to config your conda environment

## Usage
We provide a simple example here.

```python
import os
from utils.config import FilterConfig, LLMConfig
from utils.settings import settings
from utils.type import VoteType
from models import LM, RM

os.environ['OPENAI_API_BASE'] = 'your url here'
os.environ['OPENAI_API_BASE'] = 'your key here'

config = FilterConfig(type = VoteType.EQUALVOTE)
llm_config = LLMConfig(model_name = 'xxx')
settings.lm = LM(llm_config)
settings.rm = RM()

data = pd.read_csv('data/review.csv')
data.attrs['embedding_path'] = xxx
instruction = 'The {review} is Positive'
data.semantic_filter(instruction)

```

