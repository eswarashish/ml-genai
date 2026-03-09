from src.models.vanilla import VanillaLSTM
from src.core.support.torch import TorchDevice
from typing import TypedDict


class LSTMState(TypedDict):
    device: TorchDevice
    lstm: VanillaLSTM
