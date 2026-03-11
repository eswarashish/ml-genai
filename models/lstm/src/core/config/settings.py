from pydantic_settings  import BaseSettings
from pydantic import Field


class LSTM(BaseSettings):
    hidden_size: int = Field(default=24)
    dropout: float = Field(default=0.4)
    num_layers : int = Field(default=3)


settings = LSTM()