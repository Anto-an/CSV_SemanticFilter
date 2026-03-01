import models


class Settings:
    lm: models.LM | None = None
    helper_lm: models.LM | None = None
    rm: models.RM | None = None
    
    parallel_groupby_max_threads: int = 8

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")
            setattr(self, key, value)
    
    def __str__(self):
        return str(vars(self))
    
settings = Settings()