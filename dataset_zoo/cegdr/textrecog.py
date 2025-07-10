# crop every text instance to a mini-image ─> TextRecogCropPacker
data_root = 'data/CEGD-R_train_test'          # path relative to this config file
cache_path = 'data/cache'

common = dict(
    gatherer=dict(                      # gt_img_123.txt  ↔  123.jpg
        type='PairGatherer',
        img_suffixes=['.jpg', '.JPG', '.png', '.PNG'],
        rule=[r'eartags(\d+)\.(jpg|png)', r'gt_eartags\1.txt'],
        img_dir='Image',
        ann_dir='Labels',
        ),
    parser=dict(type='ICDARTxtTextDetAnnParser'),
    packer=dict(type='TextRecogCropPacker'),   # <-- crops & packs
    dumper=dict(type='JsonDumper'),
)

from copy import deepcopy

# Prepare separate configs for each split with proper image directories
train_preparer = deepcopy(common)
val_preparer   = deepcopy(common)
test_preparer  = deepcopy(common)

train_preparer['gatherer']['img_dir'] = 'Image/train'
val_preparer['gatherer']['img_dir']   = 'Image/val'
test_preparer['gatherer']['img_dir']  = 'Image/test'

config_generator = dict(type='TextRecogConfigGenerator')
