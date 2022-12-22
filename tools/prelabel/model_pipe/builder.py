from .registry import PIPELINES


def build_pipeline(cfg):
    pipeline = PIPELINES.build(cfg)
    return pipeline
