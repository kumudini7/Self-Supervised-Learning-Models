# Self-Supervised-Learning-Models
# Self-Supervised Learning for Visual Representations

This repository implements self-supervised learning on ImageNet-100 using:

- SimCLR (Contrastive Learning)
- MAE (Masked Autoencoding)

## Structure
- `models/`: Backbone and projection models
- `train/`: Pretraining and evaluation scripts
- `utils/`: Dataset and metrics

## Run Instructions
1. Clone the repo & install requirements:
    ```bash
    pip install -r requirements.txt
    ```

2. Pretrain SimCLR:
    ```bash
    python train/pretrain_simclr.py
    ```

3. Evaluate:
    ```bash
    python train/SimCLR_linear_probe.py
    ```
4. Pretrain MAE:
    ```bash
    python train/pretrain_mae.py
    ```
5. Evaluate:
    ```bash
    python train/mae_linear_probe.py
    ```
## Dataset
Use the provided Google Drive dataset and place it inside `data/`.
https://drive.google.com/file/d/1MdmkhdkhNjXM_PZDaZ9kRuoaS80vsZ8_/view?usp=sharing
## References
- SimCLR: https://arxiv.org/abs/2002.05709
- MAE: https://arxiv.org/abs/2111.06377
