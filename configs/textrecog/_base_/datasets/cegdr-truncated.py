cegdr_truncated_textrecog_data_root = 'data/CEGD-R_train_test'

cegdr_truncated_textrecog_train = dict(
    type='OCRDataset',
    data_root=cegdr_truncated_textrecog_data_root,
    ann_file='textrecog_train_truncated.json',
    pipeline=None,
    metainfo=dict(dataset_name='CEGDR-R_recog_trunc')
)

cegdr_truncated_textrecog_test = dict(
    type='OCRDataset',
    data_root=cegdr_truncated_textrecog_data_root,
    ann_file='textrecog_test_truncated.json',
    test_mode=True,
    pipeline=None,
    metainfo=dict(dataset_name='CEGDR-R_recog_trunc') 
)

# 1. import the module so the class is registered
custom_imports = dict(
    imports=['mmocr_custom.hooks.mlflow_dataset_hook'],
    allow_failed_imports=False,
)

# 2. enable the hook
custom_hooks = [dict(type='MlflowDatasetHook', priority='LOW')]