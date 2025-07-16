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