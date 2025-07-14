_base_ = [
    'mmocr::textrecog/nrtr/nrtr_resnet31-1by8-1by4_6e_st_mj.py',
    '../_base_/datasets/cegdr-truncated.py',
]

test_list = [_base_.cegdr_truncated_textrecog_test]  # type: ignore[attr-defined]

test_dataset = dict(
    type='ConcatDataset',
    datasets=test_list,
    pipeline=_base_.test_pipeline,  # type: ignore[attr-defined]
)

val_dataloader = dict(
    _delete_=True,
    batch_size=192,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset,
)

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol'],
            prefix='test'),        # <-- here
        dict(type='CharMetric', prefix='test')
    ])


test_evaluator = val_evaluator

train_list = [_base_.cegdr_truncated_textrecog_train]  # type: ignore[attr-defined]

train_dataset = dict(
    type='ConcatDataset',
    datasets=train_list,
    pipeline=_base_.train_pipeline,  # type: ignore[attr-defined]
)

train_dataloader = dict(
    _delete_=True,
    batch_size=192,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

work_dir = 'work_dirs/nrtr_custom_cegdr-truncated' 

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        tracking_uri='{{$MLFLOW_TRACKING_URI:http://localhost:5000}}',
        exp_name='mmocr_recog',
        artifact_suffix=('.json', '.log', '.py', 'yaml', '.pth'),
    ),
]

visualizer = dict(
    _delete_=True,                     # wipe the one from default_runtime
    type='TextRecogLocalVisualizer',
    vis_backends=vis_backends,         # <- **now uses your list**
    name='visualizer',
)  

auto_scale_lr = dict(base_batch_size=192) 