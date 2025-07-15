# Training MMOCR models

## Train MMOCR recognition models
### Quick smoke test on 1 epoch:
```bash
papermill mmocr_recog_cegdr_finetune_pretrained.ipynb \
  --kernel python3 /dev/null --log-output \
  -p SMOKE_TEST True \
  -p NUM_MODELS 1 \
  -p CONFIG_LIST 'nb_configs/mmocr_recog_model_list.yml'
```

```bash
tmux new-session -d -s mmocr_recog_smoke \
  "papermill mmocr_recog_cegdr_finetune_pretrained.ipynb \
     --kernel python3 /dev/null --log-output \
     -p SMOKE_TEST True \
     -p NUM_MODELS 1 \
     -p CONFIG_LIST 'nb_configs/mmocr_recog_model_list.yml'"
```

## Train all models:
```bash
papermill mmocr_recog_cegdr_finetune_pretrained.ipynb \
  --kernel python3 /dev/null --log-output \
  -p SMOKE_TEST False \
  -p NUM_MODELS None \
  -p CONFIG_LIST 'nb_configs/mmocr_recog_model_list.yml'
```

```bash
tmux new-session -d -s mmocr_recog_full \
  "papermill mmocr_recog_cegdr_finetune_pretrained.ipynb \
     --kernel python3 /dev/null --log-output \
     -p SMOKE_TEST False \
     -p NUM_MODELS None \
     -p CONFIG_LIST 'nb_configs/mmocr_recog_model_list.yml'"
```

## Train MMOCR detection models
### Quick smoke test on 1 epoch:
```bash
papermill mmocr_det_cegdr_finetune_pretrained.ipynb \
  --kernel python3 /dev/null --log-output \
  -p SMOKE_TEST True \
  -p NUM_MODELS 1 \
  -p CONFIG_LIST 'nb_configs/mmocr_det_model_list.yml'
```

```bash
tmux new-session -d -s mmocr_det_smoke \
  "papermill mmocr_det_cegdr_finetune_pretrained.ipynb \
     --kernel python3 /dev/null --log-output \
     -p SMOKE_TEST True \
     -p NUM_MODELS 1 \
     -p CONFIG_LIST 'nb_configs/mmocr_det_model_list.yml'"
```

### Train all models:
```bash
papermill mmocr_det_cegdr_finetune_pretrained.ipynb \
  --kernel python3 /dev/null --log-output \
  -p SMOKE_TEST False \
  -p NUM_MODELS None \
  -p CONFIG_LIST 'nb_configs/mmocr_det_model_list.yml'
```

```bash
tmux new-session -d -s mmocr_det_full \
  "papermill mmocr_det_cegdr_finetune_pretrained.ipynb \
     --kernel python3 /dev/null --log-output \
     -p SMOKE_TEST False \
     -p NUM_MODELS None \
     -p CONFIG_LIST 'nb_configs/mmocr_det_model_list.yml'"
```

## BONUS: train 1 det, 1 recog, then ALL det, ALL recog
```bash
sudo fuser -k /dev/nvidia* && tmux new-session -d -s mmocr_pipeline "\
  papermill mmocr_det_cegdr_finetune_pretrained.ipynb \
    --kernel python3 /dev/null --log-output \
    -p SMOKE_TEST False \
    -p NUM_MODELS 1 \
    -p CONFIG_LIST 'nb_configs/mmocr_det_model_list.yml' && \
  papermill mmocr_recog_cegdr_finetune_pretrained.ipynb \
    --kernel python3 /dev/null --log-output \
    -p SMOKE_TEST False \
    -p NUM_MODELS 1 \
    -p CONFIG_LIST 'nb_configs/mmocr_recog_model_list.yml' && \
  papermill mmocr_det_cegdr_finetune_pretrained.ipynb \
    --kernel python3 /dev/null --log-output \
    -p SMOKE_TEST False \
    -p NUM_MODELS None \
    -p CONFIG_LIST 'nb_configs/mmocr_det_model_list.yml' && \
  papermill mmocr_recog_cegdr_finetune_pretrained.ipynb \
    --kernel python3 /dev/null --log-output \
    -p SMOKE_TEST False \
    -p NUM_MODELS None \
    -p CONFIG_LIST 'nb_configs/mmocr_recog_model_list.yml' \
"
```