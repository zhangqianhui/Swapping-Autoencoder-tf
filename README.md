# swapping-autoencoder-tf
The unofficial  tensorflow implementation of Swapping Autoencoder for Deep Image Manipulation

![](img/model.png)


## Dependencies

```bash
Python=3.6
tensorflow=1.14
pip install -r requirements.txt

```
Or Using Conda

```bash
-conda create -name SAE python=3.6
-conda install tensorflow-gpu=1.14 or higher
```
Other packages installed by pip.

## Usage

- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/swapping-autoencoder-tf
cd swapping-autoencoder-tf

```

- Download the CelebAHQ dataset

  Download the tar of CelebAGaze dataset from [Google Driver Linking](https://drive.google.com/file/d/1_6f3wT72mQpu5S2K_iTkfkiXeeBcD3wn/view?usp=sharing).
  
  ```bash
  cd your_path
  tar -xvf CelebAGaze.tar
  ```
  
  Please edit the options.py and change your dataset path
  

- Train the model using command line with python

```bash
python train.py --gpu_id=0 --exper_name='log10_10_1' --data_dir='../dataset/CelebAMask-HQ/CelebA-HQ-img/'
```
- Test the model

```bash
python test.py --gpu_id=0 --exper_name='log10_10_1' --data_dir='../dataset/CelebAMask-HQ/CelebA-HQ-img/'
```

Or Using scripts for training 

```bash
bash scripts/train_log10_10_1.sh
```

For testing

```bash
bash scripts/test_log10_10_1.sh
```

## Experiment Result 

### Training results on CelebAHQ. 1st-4th colums are structure input, texture input, reconstruction, swapped

![](img/train.jpg)

### Testing results on CelebAHQ. 1st-4th colums are structure input, texture input, reconstruction, swapped

![](img/test.jpg)

