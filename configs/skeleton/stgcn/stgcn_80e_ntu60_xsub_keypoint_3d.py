model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = 'data/ntu60_xsub/train.pkl'
ann_file_val = 'data/ntu60_xsub/val.pkl'

train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PackSkeInputs')
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PackSkeInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline))

# runtime settings
work_dir = './work_dirs/stgcn_80e_ntu60_xsub_keypoint_3d/'
workflow = [('train', 1)]

custom_imports = dict(imports=['mmcv.transforms'], allow_failed_imports=False)
# 设置 registry 默认的 scope 为当前 repo，如 MMDet 中应当设置 mmdet
default_scope = 'mmaction'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

default_hooks = dict(
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5)
)

train_cfg = dict(by_epoch=True, max_epochs=80)
val_cfg = dict(interval=1)
test_cfg = dict()

param_scheduler = [
    dict(type='MultiStepLR', epoch_interval=(0, 80), by_epoch=True, milestones=[10, 50], gamma=0.1)
]

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)

log_level = 'INFO'
load_from = None
resume = False
resume_from = None