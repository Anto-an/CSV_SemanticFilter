import os
from utils.config import FilterConfig, LLMConfig
from utils.settings import settings
from utils.type import VoteType
from models import LM, RM


os.environ['OPENAI_API_BASE'] = 'http://localhost:8000/v1'
os.environ['OPENAI_API_KEY'] = 'arbitrary_key'

def read_data():
    pass

def test():
    data = read_data()
    user_instrution = 'The {review} is positive'

    result = data.semantic_filter(

    )

    config = FilterConfig(type = VoteType.EQUALVOTE)
    llm_config = LLMConfig(model_name = 'xxx')
    settings.lm = LM(llm_config)
    settings.rm = RM()

    