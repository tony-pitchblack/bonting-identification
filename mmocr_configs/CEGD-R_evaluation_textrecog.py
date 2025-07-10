_base_ = [
    'mmocr::textrecog/abinet/abinet-vision_20e_st-an_mj.py'
]

dataset_type = 'OCRDataset'          # keeps all default parsing code
data_root = 'data/CEGD-R_MMOCR/'
work_dir = 'work_dirs/CEGD-R_evaluation_textrecog'

test_evaluator = dict(
    _delete_=True,    # remove dataset_prefixes from base config
    metrics=[
        dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ]
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        _delete_=True,          # <-- wipe the old ConcatDataset fields
        type='OCRDataset',
        data_root=data_root,    # optional, if you keep data_root outside
        ann_file='annotations/test.json',  # folder that has gt_*.txt test_mode=True,
        test_mode=True,
        pipeline=_base_.test_pipeline,   # re-use the original pipeline

    ))