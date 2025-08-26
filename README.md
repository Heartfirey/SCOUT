<div align="center">
<img width="96" height="96" alt="image" src="https://github.com/user-attachments/assets/30fbc474-982f-4799-8c62-da4ccfe5439d" />
<h1>SCOUT: Semi-supervised Camouflaged Object Detection by Utilizing Text and Adaptive Data Selection</h1>

<a href="https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/392.pdf" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Paper-SCOUT" alt="Paper PDF"></a>
<a href="https://arxiv.org/abs/2508.17843"><img src="https://img.shields.io/badge/arXiv-2508.17843-b31b1b" alt="arXiv"></a>
<a href="https://heartfirey.top/project_page/SCOUT/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

**[Key Laboratory of Multimedia Trusted Perception and Effecient Computing](https://multimedia.xmu.edu.cn)**

[Weiqi Yan](https://heartfirey.top), [Lvhai Chen](https://jormungand00222.github.io/Jormungand00222/), [Shengchuan Zhang](), [Liujuan Cao]()

</div>

## Updates

- [August 26, 2025] Code and dataset annotations released.

- [April 29, 2025] The paper has been accepted by IJCAI 2025.

## Abstract
> The difficulty of pixel-level annotation has significantly hindered the development of the Camouflaged Object Detection (COD) field. To save on annotation costs, previous works leverage the semi-supervised COD framework that relies on a small number of labeled data and a large volume of unlabeled data. We argue that there is still significant room for improvement in the effective utilization of unlabeled data. To this end, we introduce a Semi-supervised Camouflaged Object Detection by Utilizing Text and Adaptive Data Selection (SCOUT). It includes an Adaptive Data Augment and Selection (ADAS) module and a Text Fusion Module (TFM). The ADSA module selects valuable data for annotation through an adversarial augment and sampling strategy. The TFM module further leverages the selected valuable data by combining camouflage-related knowledge and text-visual interaction. To adapt to this work, we build a new dataset, namely RefTextCOD. Extensive experiments show that the proposed method surpasses previous semi-supervised methods in the COD field and achieves state-of-the-art performance.

## Datasets / Annotations



## Quick Start

### Environment Setup

```
conda create -n scout python==3.9 -y && conda activate scout
pip install -r requirement.txt
```

Then install [FlashAttention](https://github.com/Dao-AILab/flash-attention)

```
git clone https://github.com/ROCm/flash-attention.git && cd flash-attention
python setup.py install
```

### Dataset Preparation

Downloading dataset: [GoogleDrive](https://drive.google.com/drive/folders/19MaIVAcqr8sIv0R1hIq7MZhPqO-9_s8v?usp=drive_link) and moving it into `./dataset`.

### Training

```
bash ./scripts/train.sh 0,1,2,3,4,5,6,7 config/runs/scout.py
```

### Evaluation

Downloading checkpoints at GoogleDrive(https://drive.google.com/drive/folders/1qkphaFNKYGs-y9w1uIrtuUZ3oV8gifZP?usp=drive_link).
“fixed” means testing with fixed text, while “accurate” means testing with directive text.

```
bash ./scripts/test.sh 0 config/runs/scout.py weights/split0.01.pth fixed
```



### Adaptive Data Selection

```
bash ./scripts/launch_select.sh 0 config/runs/scout.py weights/split0.01.pth 
```

## Acknowledgements

This work was supported by the National Science Fund for Distinguished Young Scholars (No. 62025603), the National Natural Science Foundation of China (No. U21B2037, No. U22B2051, No. U23A20383, No. 62176222, No. 62176223, No. 62176226, No. 62072386, No. 62072387, No. 62072389, No. 62002305 and No. 62272401), and the Natural Science Foundation of Fujian Province of China (No. 2021J06003, No. 2022J06001).

## Reference

```bibtex
@InProceedings{yan2025scout,
    author    = {Yan, Weiqi and Chen, Lvhai and Zhang, Shengchuan and Zhang, Yan and Cao, Liujuan},  
    title     = {SCOUT: Semi-supervised Camouflaged Object Detection by Utilizing Text and Adaptive Data Selection},
    booktitle = {The 34th International Joint Conference on Artificial Intelligence},
    year      = {2025}
}
```
