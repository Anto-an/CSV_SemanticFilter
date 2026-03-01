import pandas as pd
from typing import Any

from utils.config import FilterConfig
from operators.filter.filter_func import semantic_filter

@pd.api.extensions.register_dataframe_accessor("semantic_filter")
class SemanticFilterDataFrame:
    
    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")
    
    def __call__(
        self,
        user_instruction: str,
        config: FilterConfig
    ) -> pd.DataFrame:
        return semantic_filter(self._obj, user_instruction, config)