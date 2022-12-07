from mmpose.registry import HOOKS
from mmengine.hooks.hook import Hook
from mmpose.models.backbones.utils.repvgg import repvgg_model_convert


@HOOKS.register_module()
class RepVGGHook(Hook):

    def before_test_epoch(self, runner) -> None:
        runner.model = repvgg_model_convert(runner.model)
