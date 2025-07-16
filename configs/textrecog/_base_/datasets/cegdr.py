cegdr_textrecog_data_root = 'data/CEGD-R_train_test'

cegdr_textrecog_train = dict(
    type='OCRDataset',
    data_root=cegdr_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None,
    metainfo=dict(dataset_name='CEGDR-R_recog')
)

cegdr_textrecog_test = dict(
    type='OCRDataset',
    data_root=cegdr_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None,
    metainfo=dict(dataset_name='CEGDR-R_recog')
)