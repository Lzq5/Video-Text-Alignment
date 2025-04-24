# Multi-Sentence Grounding for Long-term Instructional Video

Zeqian Li, Qirui Chen, Tengda Han, Ya Zhang, Yanfeng Wang, Weidi Xie

[[project page]](https://lzq5.github.io/Video-Text-Alignment/)
[[Arxiv]](https://arxiv.org/abs/2312.14055)
[[Dataset]](https://huggingface.co/datasets/zeqianli/HowToStep)

## Environments

```bash
conda create --name align python=3.10.0
conda activate align
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
wandb login ***
```

## Dataset Preparation
* [**HowToStep**](https://huggingface.co/datasets/zeqianli/HowToStep): An automatically generated dataset that transforms ASR transcripts into descriptive steps by prompting the LLM and then aligns steps to the video through a two-stage determination procedure.
* [**HowTo100M**](https://github.com/TengdaHan/TemporalAlignNet/tree/main/htm_zoo): WhisperX ASR output and InternVideo & CLIP-L14 visual features for HowTo100M.
* [**HTM-Align**](https://github.com/TengdaHan/TemporalAlignNet/tree/main/htm_align): A manually annotated 80-video subset for narration alignment evaluation.
* [**HT-Step**](https://eval.ai/web/challenges/challenge-page/2082/submission): A collection of temporal annotations on videos from the HowTo100M dataset for procedural step grounding evaluation.

## Code Overview
Some of the main components are
* [[./configs]](./configs/): Configs for training or inference.
* [[./src/data]](./src/data/): Scripts for feature extraction.
* [[./src/dataset]](./src/dataset/): Data loader.
* [[./src/model]](./src/model/): Our main model with all its building blocks.
* [[./src/trainer]](./src/trainer/): Our code for training and inference.
* [[./src/utils]](./src/utils/): Utility functions for training, inference, and visualization.

## Feature Extraction
In [[data]](./src/data/), we provide the feature extraction script for extracting visual features and textual features using [InternVideo-MM-L14](https://github.com/OpenGVLab/InternVideo). 

Modify the paths of raw video, HowTo100M, HTM-Align, HT-step in the file, and the visual or textual features can be extracted via:
```
python ./src/data/extract_textualfeature.py
python ./src/data/extract_visualfeature.py
```

## Training
Modify the paths in [`dataset.py`](./src/dataset/dataset.py), mainly the paths for visual/textual features and the annotation files. For the procedural step grounding task, we set 'text_shuffle' to True and 'text_pe' to False in [`htm.yaml`](./configs/htm.yaml); while for the narration alignment task, we set 'text_shuffle' to False and 'text_pe' to True.

The training command is:
```
python main.py --gpu 0 --config_file configs/htm.yaml --run_name train
```

## Inference
During inference, only need to modify the 'checkpoint' in [`htm.yaml`](./configs/htm.yaml) to the path of the trained model. The settings for 'text_shuffle' and 'text_pe' are the same as during training.

The inference command is:
```
python main.py --gpu 0 --config_file configs/htm.yaml --run_name eval
```

**[Optional] Evaluating Our Pre-trained Model**

We also provide pre-trained models for HTM-Align and HT-Step. The model with all training logs can be downloaded from [NSVA](https://huggingface.co/zeqianli/HowToStep-NSVA). 

The results should be

| Model Name | Task | Evaluation Dataset | Performance |
|-------------------|-------|-------|-------|
| NSVA_narration.pth | Narration Alignment| HTM-Align | 69.8 |
| NSVA_step.pth | Procedural Step Grounding | HT-Step | 47.0 |

## Citation
If you are using our code, please consider citing our paper.
```bibtex
@inproceedings{li2024multi,
  title={Multi-Sentence Grounding for Long-term Instructional Video},
  author={Li, Zeqian and Chen, Qirui and Han, Tengda and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  booktitle={European Conference on Computer Vision},
  pages={200--216},
  year={2024},
  organization={Springer}
}
```

## Contact
If you have any question, please feel free to contact lzq0103@sjtu.edu.cn.
