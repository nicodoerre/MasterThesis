As a first step, please set up the required environment by running `conda env create -f env.yml`.
# Data Generation
The data generation process is designed in a modular fashion. At the heart of the process lies the `constants.py` file, which contains all the randomizable properties for the individua graphs. The `graphing_functions.py` and the `graphing_functions_advanced.py` files contain the functions for creating the individual graphs. The `generate_data.py` script is the main script and loops over all the functions a specified number of times. Each time the function gets called, it does so with a new set of parameters. The created plots are saved in a specified folder. To create a train/validation/test split, simply run the `create_dataset.py` file. Additionally, a example dataset can be downloaded [here](https://huggingface.co/datasets/ndoerre/graphs_and_plots).

# Models
## EDSR
The code for the EDSR model of ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)is taken from the [official PyTorch implementation](https://github.com/sanghyun-son/EDSR-PyTorch) on GitHub. The final model was trained by running a hyperparameter search on x2 scale. Then the full model on x2 was trained. This fully trainde x2 model served as a basis for the hyperparameter searches for the higher scales. The fully trained higher scale models were then obtained by using the found hyperparameters of the respective scale. To run the hyperparameter search, simply call the `search_hyper_params` script. The search was only done on a small subset of training data. The EDSR training expects the low resolution counterparts of the images as a seperate set, so the `generate_lr.py` script takes care of that. For specific details to train an EDSR model, please refer to the official repository. The custom data handling is done in the `diagrams` files of the `data` subfolder in the `scr` folder.

To train a model, either run  `python main.py --template EDSR_paper --scale 2 --patch_size 96 --n_resblocks 32 --n_feats 256 --test_every 300 --batch_size 16 --lr 0.0001 --res_scale 0.1 --epochs 300 --data_train DIAGRAMS --data_test DIAGRAMS --dir_data path/to/data --ext sep --save path/to/save --reset`, or with the additional `--pre_train` flag when training higher scale models. This loads the model weights of an fully trained x2 model. For details on the specific flags, please see the [official code](https://github.com/sanghyun-son/EDSR-PyTorch). Please note, that the repository expects the following folder structure:  
```
<dir_data>/DIAGRAMS/
        train/HR/              *.png  (RGB, HR originals)
        train/LR_bicubic/X2/   *.png
        train/LR_bicubic/X3/   *.png
        train/LR_bicubic/X4/   *.png
        val/HR/                …
        val/LR_bicubic/X2/…X4/ …

```

## WaveMixSR
To train the model on custom data, simply run the `train_div2k.py` file like `python train_div2k.py -x 2 -metric psnr`, whereas the `x` flag indicates the scale and the `metric` flag the target metric to optimize. Additionally, the `DATA_ROOT` variable needs to be set accordingly. The custom verion of this script is for custom loss functions.

The code of the model and the overall training procedure is taken from the [official PyTorch implementation](https://github.com/pranavphoenix/WaveMixSR/tree/main) of the authors and it adheres to the paper ["WaveMixSR: Resource-Efficient Neural Network for Image Super-Resolution"](https://openaccess.thecvf.com/content/WACV2024/html/Jeevan_WaveMixSR_Resource-Efficient_Neural_Network_for_Image_Super-Resolution_WACV_2024_paper.html). Also here, a additional hyperparameter search was conducted in the `search_hyper_params` script. in order to utilize "offline" data, a custom dataset class was implemented.

## LIIF
The code of [LIIF](https://arxiv.org/abs/2012.09161) is from the [official github repository](https://github.com/yinboc/liif) of the authors. In order to train LIIF, the config file `train_custom.yaml` needs to be adjusted. It must conatin a valid path to the root of the training and validation dataset. After that, the training can be started by running `python train_liif.py --config configs/train_custom.yaml --gpu 0`.

# Sharpness Measure and LPIPS
The Sharpness Measure is implemented according to ["Image Sharpness Measure for Blurred Images in Frequency Domain"](https://www.sciencedirect.com/science/article/pii/S1877705813016007). The algorithm is located in the `fm_score.py` file in the `sharpness_measure` folder. The veriosn of [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) is taken from the [official PyTorch Repository](https://github.com/richzhang/PerceptualSimilarity) of the authors.

# Evaluation Pipeline
The `OCR_metric` folder conatins the main evaluation pipeline, as well as ways to analyse the produced model output. The `analysis.ipynb` notebook plots and saves FFT as well as Power Spectra of specific images. For this, simply follow the instructions in the notebook. The `get_imgs` folder contains the model files as well as scripts to produce the super resolved outputs. The `generate_img.py` script produces such outputs by selecting the desired scale and the model. Note, that if LIIF should be used, the script needs the demo file of the official [LIIF Repository](https://github.com/yinboc/liif). The main pipeline can be tested in the `main_pipeline.ipynb` notebook. The pipeline leverages [Tesseract 4] (https://github.com/tesseract-ocr/tesseract) as OCR engine. The code of the pipeline itself is located in the `pipeline.py` file.