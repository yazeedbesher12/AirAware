from .sequence_model import TorchSequenceModel


class LSTMModel(TorchSequenceModel):
    def __init__(self):
        super().__init__("lstm")
