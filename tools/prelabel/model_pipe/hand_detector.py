from .registry import COMPUTE_NODES
import onnxruntime


@COMPUTE_NODES.register_module()
class HandDetector:

    def __init__(self, model_path) -> None:
        self.model = onnxruntime.InferenceSession(model_path)

    def process(self, data):
        return data
