_base_ = [
    'mmocr::textrecog/abinet/abinet_20e_st-an_mj.py',
    '../_base_/datasets/cegdr-truncated.py',
]

# ---- TEST / VAL -------------------------------------------------------------------------

val_dataloader = dict(
    _delete_=True,
    batch_size=96,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # Re-use the dataset from the base cfg and just attach the test pipeline
    dataset={
        **_base_.cegdr_truncated_textrecog_test,  # type: ignore[attr-defined]
        'pipeline': _base_.test_pipeline,          # type: ignore[attr-defined]
    },
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

# ---- TRAIN ------------------------------------------------------------------------------

train_dataloader = dict(
    _delete_=True,
    batch_size=96,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # Re-use the dataset from the base cfg and attach the training pipeline
    dataset={
        **_base_.cegdr_truncated_textrecog_train,  # type: ignore[attr-defined]
        'pipeline': _base_.train_pipeline,          # type: ignore[attr-defined]
    },
)

work_dir = 'work_dirs/abinet_custom_cegdr-truncated' 

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

auto_scale_lr = dict(base_batch_size=96 * 8) 