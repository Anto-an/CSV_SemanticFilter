import pandas as pd
from typing import Any

from utils.config import JoinConfig
from operators.join.join_func import semantic_join

@pd.api.extensions.register_dataframe_accessor("semantic_join")
class SemanticJoinDataFrame:
    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")
    
    def __call__(
        self,
        right: pd.DataFrame,
        user_instruction: str,
        config: JoinConfig
    ) -> pd.DataFrame:
        return semantic_join(self._obj, right, user_instruction, config)