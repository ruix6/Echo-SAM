# Echo-SAM
This is the official repository for Echo-SAM.


## Installation
1. Create a virtual environment `conda create -n echosam python=3.10 -y` and activate it `conda activate echosam`
2. `git clone https://github.com/ruix6/Echo-SAM`
3. Enter the Echo-SAM folder `cd Echo-SAM` and run `pip install -r requirements.txt`


## Get Started
Download the [model checkpoint](https://drive.google.com/file/d/12XbNDJaC_QovdXaVTT8tAPvBhsadLA22/view?usp=sharing) and place it at e.g., `work_dir/Echo_SAM/Echo_SAM_02100126.pth`

We provide gui for testing the model on your images

```bash
python gui_point.py
```

Load the image to the GUI and specify segmentation targets by pointing LA, LV or MYO.


## Model Training

### Data preprocessing

Download [MedSAM checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at `work_dir/MedSAM/medsam_vit_b.pth` .

Download dataset for training.

### Training and Testing

```bash
python echo_sam_train.py
```
```bash
python echo_sam_test.py
```



## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We thank bowang-lab for making the source code of [MedSAM](https://github.com/bowang-lab/MedSAM)  publicly available.
- We also thank Xian Lin for making the source code of [SAMUS](https://github.com/xianlin7/SAMUS)  publicly available.
