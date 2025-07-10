cegdr_data_root = 'data/CEGD-R_MMOCR/'

cegdr_textrecog_test = dict(
    type='OCRDataset',
    data_root=cegdr_data_root,
    ann_file='annotations/test.json',
    test_mode=True,
    pipeline=None,
) 