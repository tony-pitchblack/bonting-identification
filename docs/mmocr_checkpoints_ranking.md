# MMOCR – Recommended Checkpoints for Unseen Tasks

**Text Detection**  
1. **DBNet++ ResNet-50 + DCNv2 (FPNC, 1200e, ICDAR 2015)** – best average hmean and strong transfer thanks to Adaptive Scale Fusion and deformable convolutions.  
2. **TextSnake ResNet-50-oCLIP (FPN-UNet, 1200e, CTW1500)** – excels on curved / long texts; oCLIP pre-training boosts generalisation.  
3. **FCENet ResNet-50 + DCNv2 (FPN, 1500e, CTW1500)** – Fourier contour head captures arbitrary-shaped instances, good zero-shot.  
4. **PSENet ResNet-50-oCLIP (FPNF, 600e, ICDAR 2015)** – progressive scale expansion handles densely packed text.  
5. **PANet ResNet-18 (FPEM-FFM, 600e, ICDAR 2017)** – lightweight yet robust when resources are limited.

Config paths: `configs/textdet/**` in MMOCR repo; each has matching weight URL in its `metafile.yml`.

---

**Text Recognition**  
1. **ABINet-Vision (20e, SynthText + MJ + ST-AN)** – iterative language correction yields top accuracy and resilience to noise.  
2. **SVTR-Base (5e, SynthText + MJ)** – pure vision transformer; strong at complex fonts and open vocabulary.  
3. **SAR ResNet-31 (parallel decoder, 6e, MJ + ST)** – attention decoder handles variable-length strings well; good multilingual transfer.  
4. **RobustScanner ResNet-45 (5e, MJ)** – fuses 2-D and sequence clues, stable on distorted or low-resolution text.  
5. **CRNN ResNet-34 (2-5x, MJ)** – classic baseline; fastest on CPU and still competitive for simple scripts.

Config paths: `configs/textrecog/**`; use weight links given in each `metafile.yml`. 