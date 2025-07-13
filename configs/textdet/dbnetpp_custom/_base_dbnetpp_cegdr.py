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

# Use the shared CEGDR test dataset and attach our pipeline

test_dataset = dict(
    type='ConcatDataset',
    datasets=[_base_.cegdr_textdet_test],  # type: ignore[attr-defined]
    pipeline=test_pipeline,
)

# Dataloaders

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset,
)

test_dataloader = val_dataloader

# Evaluator

val_evaluator = dict(
    _delete_=True,
    type='HmeanIOUMetric',
    prefix='cegdr',
)

test_evaluator = val_evaluator 