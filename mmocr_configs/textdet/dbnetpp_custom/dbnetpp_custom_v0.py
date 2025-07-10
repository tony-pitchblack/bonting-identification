_base_ = [
    'mmocr::textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

# Define the test pipeline explicitly (copied from DBNet++ base config)
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(type='LoadOCRAnnotations', with_polygon=True, with_bbox=True, with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Custom dataset configuration for CEGD-R text detection evaluation
cegdr_test = dict(
    type='OCRDataset',
    data_root='data/CEGD-R_MMOCR/',
    ann_file='annotations/test.json',
    test_mode=True,
    pipeline=test_pipeline,
)

# DataLoader for evaluation
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=cegdr_test,
)

test_dataloader = val_dataloader

# Evaluator settings for text detection
val_evaluator = dict(
    _delete_=True,
    type='HmeanIOUMetric',
    prefix='cegdr'
)

test_evaluator = val_evaluator

# Working directory for outputs
work_dir = 'work_dirs/dbnetpp_custom_cegdr' 