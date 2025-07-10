# Datasets management

## Create MMOCR datasets
### CEGD-R (recognition)
```bash
./manage_data/download_CEGD-R.sh
python ./manage_data/split_CEGD-R.py
python mmocr/tools/dataset_converters/prepare_dataset.py \
    cegdr \
    --task textrecog \
    --dataset-zoo-path dataset_zoo \
    --splits train test \
    --nproc 1
```