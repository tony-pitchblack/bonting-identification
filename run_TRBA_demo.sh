CUDA_VISIBLE_DEVICES=0 python3 ./dl-ocr-bench/dl_ocr_bench/demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder data/bonting-identification/samples/ear-tags/recognition \
--saved_model ckpt/pretrained/TPS-ResNet-BiLSTM-Attn.pth