_base_ = ['./_base_satrn_cegdr.py']

dictionary = dict(
    type='Dictionary',
    with_unknown=True,
)

model = dict(decoder=dict(dictionary=dictionary))

work_dir = 'work_dirs/satrn_custom_cegdr_dict_unk' 