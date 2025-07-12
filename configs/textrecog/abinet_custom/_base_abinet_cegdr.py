_base_ = [
    'mmocr::textrecog/abinet/abinet_20e_st-an_mj.py',
    '../_base_/datasets/cegdr.py',
]

# Dataset settings based on the shared CEGDR definition

test_list = [_base_.cegdr_textrecog_test]  # type: ignore[attr-defined]

test_dataset = dict(
    type='ConcatDataset',
    datasets=test_list,
    pipeline=_base_.test_pipeline,  # type: ignore[attr-defined]
)

# DataLoader for evaluation
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset,
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

# Dataset settings for training
train_list = [_base_.cegdr_textrecog_train]  # type: ignore[attr-defined]

train_dataset = dict(
    type='ConcatDataset',
    datasets=train_list,
    pipeline=_base_.train_pipeline,  # type: ignore[attr-defined]
)

# DataLoader for training
train_dataloader = dict(
    _delete_=True,
    batch_size=96,
    num_workers=18,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

# Working directory for outputs
work_dir = 'work_dirs/abinet_custom_cegdr' 