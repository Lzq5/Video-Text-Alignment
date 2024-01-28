# Alignment

## Environments

```bash
conda create --name align python=3.10.0
conda activate align
# pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
wandb login ***
```

## Running

```bash
python main.py --gpu 0 --run_name tmp
```