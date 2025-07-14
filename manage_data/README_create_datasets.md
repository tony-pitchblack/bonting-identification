# Create datasets

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
python mmocr/tools/dataset_converters/prepare_dataset.py \
    cegdr \
    --task textdet \
    --dataset-zoo-path dataset_zoo \
    --splits train test \
    --nproc 1

### CEGD-R truncated (recognition) 
After preparing the regular train/test splits as above, you can filter the
recognition annotations so that only samples containing characters present in
`mmocr/dicts/lower_english_digits_space.txt` are kept. This significantly
reduces noise introduced by uncommon symbols.

```bash
python ./manage_data/truncate_CEGD-R_recog.py
```