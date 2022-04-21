from mmengine.registry import METRICS as MMEngine_METRICS
from mmengine.registry import Registry
from mmengine import DefaultScope
default_scope = DefaultScope.get_instance('mmengine', scope_name='mmaction')

METRICS = Registry('metrics', parent=MMEngine_METRICS)