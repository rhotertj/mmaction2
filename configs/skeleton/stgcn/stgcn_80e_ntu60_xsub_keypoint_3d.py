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
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
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

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 50])
total_epochs = 80
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stgcn_80e_ntu60_xsub_keypoint_3d/'
load_from = None
resume_from = None
workflow = [('train', 1)]
