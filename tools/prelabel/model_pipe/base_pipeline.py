from .registry import PIPELINES, COMPUTE_NODES


@PIPELINES.register_module()
class PipelineBase:

    def __init__(self, nodes, device='cpu') -> None:
        self.device = device
        self.nodes = [COMPUTE_NODES.build(node_cfg) for node_cfg in nodes]

    def run(self, data):
        for node in self.nodes:
            data = node.process(data)
        return data
