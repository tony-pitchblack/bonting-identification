_base_ = ['./_base_abinet-vision_cegdr.py']

dictionary = dict(
    type='Dictionary',
    with_unknown=True,
)

model = dict(decoder=dict(dictionary=dictionary))

work_dir = 'work_dirs/abinet-vision_custom_cegdr_dict_unk' 