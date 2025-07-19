_base_ = [
    'mmocr::textdet/panet/panet_resnet50_fpem-ffm_600e_icdar2017.py',
    '../_base_/datasets/cegdr.py',  # shared dataset definition
]

# Test pipeline (copied from DBNet++ base config)

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(type='LoadOCRAnnotations', with_polygon=True, with_bbox=True, with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
    ),
]

# ---- TRAIN ------------------------------------------------------------------------------

train_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # Re-use the dataset from the base cfg and attach the training pipeline
    dataset={
        **_base_.cegdr_textdet_train,  # type: ignore[attr-defined]
        'pipeline': _base_.train_pipeline,          # type: ignore[attr-defined]
    },
)

# ---- TEST / VAL -------------------------------------------------------------------------

val_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # Re-use the dataset from the base cfg and just attach the test pipeline
    dataset={
        **_base_.cegdr_textdet_test,  # type: ignore[attr-defined]
        'pipeline': test_pipeline,          # type: ignore[attr-defined]
    },
)

test_dataloader = val_dataloader

# Evaluators

val_evaluator = dict(
    _delete_=True,
    type='HmeanIOUMetric',
    prefix='test',
)

test_evaluator = val_evaluator

work_dir = 'work_dirs/panet_custom_cegdr'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        tracking_uri='{{$MLFLOW_TRACKING_URI:http://localhost:5000}}',
        exp_name='mmocr_det',
        artifact_suffix=('.json', '.log', '.py', 'yaml', '.pth'),
    ),
]

visualizer = dict(
    _delete_=True,                     # wipe the one from default_runtime
    type='TextDetLocalVisualizer',
    vis_backends=vis_backends,         # <- **now uses your list**
    name='visualizer',
) 

auto_scale_lr = dict(base_batch_size=16) 

# --- Fine-tuning schedule --------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1) 

# 1. import the module so the class is registered
custom_imports = dict(
    imports=[
        'mmocr_custom.hooks.mlflow_dataset_hook',
        'mmocr_custom.hooks.mlflow_checkpoint_hook',
    ],
    allow_failed_imports=False,
)

# --- Early stopping ----------------------------------------------------------
custom_hooks = [
    dict(
        type='MlflowCheckpointHook',
        interval=1,
        save_best='test/hmean',
        rule='greater',
        tracking_uri='{{$MLFLOW_TRACKING_URI:http://localhost:5000}}',
        exp_name='mmocr_det',
    ),
    dict(type='MlflowDatasetHook', priority='LOW'),
    dict(
        type='EarlyStoppingHook',
        monitor='test/hmean',   # <-- metric key to watch
        rule='greater',            # 'greater' if higher is better, 'less' otherwise
        patience=5,                # stop after 5 val epochs with no improvement
        min_delta=0.01            # a change smaller than this counts as “no improvement”
    ),
]
# Tell the checkpoint hook to keep the best model of the same metric
default_hooks = dict(
    checkpoint=None
)