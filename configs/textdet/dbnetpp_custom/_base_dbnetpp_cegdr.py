_base_ = [
    'mmocr::textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
    '../_base_/datasets/cegdr.py',  # shared dataset definition
]

# Test pipeline (copied from DBNet++ base config)

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(type='LoadOCRAnnotations', with_polygon=True, with_bbox=True, with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# ---- TRAIN ------------------------------------------------------------------------------

train_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=8,
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
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # Re-use the dataset from the base cfg and just attach the test pipeline
    dataset={
        **_base_.cegdr_textdet_test,  # type: ignore[attr-defined]
        'pipeline': test_pipeline,          # type: ignore[attr-defined]
    },
)

test_dataloader = val_dataloader

# Evaluator

val_evaluator = dict(
    _delete_=True,
    type='HmeanIOUMetric',
    prefix='test',
)

test_evaluator = val_evaluator

work_dir = 'work_dirs/dbnetpp_custom_cegdr'

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