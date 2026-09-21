from .sequence_model import TorchSequenceModel


class GRUModel(TorchSequenceModel):
    def __init__(self):
        super().__init__("gru")
