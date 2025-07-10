_base_ = [
    'mmocr::textrecog/abinet/abinet-vision_20e_st-an_mj.py',
]

# Define the test pipeline explicitly (copied from ABINet base config)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(128, 32)),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio')
    )
]

# Custom dataset configuration for CEGD-R evaluation
cegdr_textrecog_test = dict(
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
    dataset=cegdr_textrecog_test,
)

test_dataloader = val_dataloader

# Evaluator settings
val_evaluator = dict(
    _delete_=True,
    metrics=[
        dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric'),
    ]
)

test_evaluator = val_evaluator

# Working directory for outputs
work_dir = 'work_dirs/abinet_vision_custom_cegdr' 