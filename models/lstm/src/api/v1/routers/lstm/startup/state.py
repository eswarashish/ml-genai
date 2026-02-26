from src.models.lstm import ETDLSTM
from src.core.support.torch import TorchDevice
from typing import TypedDict


class LSTMState(TypedDict):
    device: TorchDevice
    lstm: ETDLSTM
