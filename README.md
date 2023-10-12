## Introduction

**This is the official PyTorch implementation of the ICCV 2023 paper.**

[UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition.pdf]()

[Supplementary.pdf]()


## Get started

1. **Prepare dataset**

    Download [CASIA-Webface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing) preprocessed by [insightface](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).
    ```console
    unzip faces_webface_112x112.zip
    ```

2. **Train model**

    Modify the 'data_path' in [train.py](train.py) (Line 56)

    Select and uncomment the 'sample_to_sample_loss' in [backbone.py](backbone.py) (Line 71)
    ```console
    python train.py
    ```

4. **Test model**
    ```console
    python pytorch2onnx.py
    zip model.zip model.onnx
    ```
    Upload model.zip to [MFR Ongoing](http://iccv21-mfr.com/#/leaderboard/academic) and then wait for the results.

    We provide a pre-trained model (ResNet-50) on [Google Drive](https://drive.google.com/file/d/167zN2NYowc6UyP4CjwPfgW3xM86oUrWD/view?usp=drive_link) for easy and direct development. This model is trained on CASIA-WebFace and achieved 50.25% on MR-All and 99.53% on LFW.

## Citation

If you find **UniTSFace** useful in your research, please consider to cite:

  ```bibtex
  @InProceedings{NeurIPS_2023,
    author    = {Zhou, Jiancan and Jia, Xi and Li, Qiufu and Shen, Linlin and Duan, Jinming},
    title     = {UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition}
  }
  ```
