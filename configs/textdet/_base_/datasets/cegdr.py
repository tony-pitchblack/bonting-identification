cegdr_textdet_data_root = 'data/CEGD-R_train_test'

cegdr_textdet_train = dict(
    type='OCRDataset',
    data_root=cegdr_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    metainfo=dict(dataset_name='CEGDR-R_det')
)

cegdr_textdet_test = dict(
    type='OCRDataset',
    data_root=cegdr_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None,
    metainfo=dict(dataset_name='CEGDR-R_det')
)

# 1. import the module so the class is registered
custom_imports = dict(
    imports=['mmocr_custom.hooks.mlflow_dataset_hook'],
    allow_failed_imports=False,
)

# 2. enable the hook
custom_hooks = [dict(type='MlflowDatasetHook', priority='LOW')]