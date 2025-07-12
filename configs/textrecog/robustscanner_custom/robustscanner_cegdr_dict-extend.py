_base_ = ['./_base_robustscanner_cegdr.py']

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../mmocr/dicts/english_digits_symbols_space.txt',
    with_unknown=False,
)

model = dict(decoder=dict(dictionary=dictionary))

work_dir = 'work_dirs/robustscanner_custom_cegdr_dict_extend' 