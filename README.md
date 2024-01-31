![image](https://github.com/filipbasara0/relic/assets/29043871/130c3459-08fc-49c1-a922-43576f2a255c)

# ReLIC

A PyTorch implementation of a computer vision self-supervised learning method based on [Representation Learning via Invariant Causal Mechanisms (ReLIC)](https://arxiv.org/abs/2010.07922).

This simple approach is very similar to [BYOL](https://arxiv.org/abs/2006.07733) and [SimCLR](https://arxiv.org/abs/2002.05709). The training technique uses a online and target encoder (EMA) with a simple critic MLP projector, while the instance discrimination loss function resembles the contrastive loss used in SimCLR. The other half of the loss function acts as a regularizer - it includes an invariance penalty, which forces the representations to stay invariant under data augmentations and amplifies intra-class distances.

![image](https://github.com/filipbasara0/relic/assets/29043871/70ccdb40-3343-4ea7-946b-80bdc1e7b85d)

Also has an experimental support for the sigmoid pairwise loss, from the [SigLIP](https://arxiv.org/abs/2303.15343) paper. This loss is generally less stable and gives slightly worse metrics, but still yields very good representations.

# Results

Models are pretrained on training subsets - for `CIFAR10` 50,000 and for `STL10` 100,000 images. For evaluation, I trained and tested LogisticRegression on frozen features from:
1. `CIFAR10` - 50,000 train images on ReLIC
2. `STL10` - features were learned on 100k unlabeled images. LogReg was trained on 5k train images and evaluated on 8k test images.

Linear probing was used for evaluating on features extracted from encoders using the scikit LogisticRegression model.

More detailed evaluation steps and results for [CIFAR10](https://github.com/filipbasara0/relic/blob/main/notebooks/linear-probing-cifar.ipynb) and [STL10](https://github.com/filipbasara0/relic/blob/main/notebooks/linear-probing-stl.ipynb) can be found in the notebooks directory. 

| Evaulation model    | Dataset | Feature Extractor| Encoder   | Feature dim | Projection Head dim | Epochs | Top1 % |
|---------------------|---------|------------------|-----------|-------------|---------------------|--------|--------|
| LogisticRegression  | CIFAR10 | ReLIC            | ResNet-18 | 512         | 64                  | 100    | 71.07  |
| LogisticRegression  | STL10   | ReLIC            | ResNet-18 | 512         | 64                  | 100    | 76.10  |
| LogisticRegression  | STL10   | ReLIC            | ResNet-50 | 2048        | 64                  | 100    | 80.40  |

[Here](https://drive.google.com/file/d/1XaZBdvPGPh2nQzzHAJ_oL41c1f8Lc_FN/view?usp=sharing) is a link to a resnet18 encoder trained on the ImageNet-1k subset. This models pefroms better on both CIFAR10 and STL10.

# Usage

### Instalation

```bash
$ pip install relic-pytorch
```

Code currently supports ResNet18, ResNet50 and an experimental version of the EfficientNet model. Supported datasets are STL10 and CIFAR10.

All training is done from scratch.

### Examples
`CIFAR10` ResNet-18 model was trained with this command:

`relic_train --dataset_name "cifar10" --encoder_model_name resnet18 --fp16_precision --gamma 0.99 --alpha 1.0`

`STL10` ResNet-50 model was trained with this command:

`relic_train --dataset_name "stl10" --encoder_model_name resnet50 --fp16_precision`

### Detailed options
Once the code is setup, run the following command with optinos listed below:
`relic_train [args...]⬇️`

```
ReLIC

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path where datasets will be saved
  --dataset_name {stl10,cifar10}
                        Dataset name
  -m {resnet18,resnet50,efficientnet}, --encoder_model_name {resnet18,resnet50,efficientnet}
                        model architecture: resnet18, resnet50 or efficientnet (default: resnet18)
  -save_model_dir SAVE_MODEL_DIR
                        Path where models
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  --fp16_precision      Whether to use 16-bit precision GPU training.
  --proj_out_dim PROJ_OUT_DIM
                        Projector MLP out dimension
  --proj_hidden_dim PROJ_HIDDEN_DIM
                        Projector MLP hidden dimension
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Log every n steps
  --gamma GAMMA         Initial EMA coefficient
  --use_siglip          Whether to use siglip loss.
  --alpha ALPHA         Regularization loss factor
  --update_gamma_after_step UPDATE_GAMMA_AFTER_STEP
                        Update EMA gamma after this step
  --update_gamma_every_n_steps UPDATE_GAMMA_EVERY_N_STEPS
                        Update EMA gamma after this many steps
```

# Citation

```
@misc{mitrovic2020representation,
      title={Representation Learning via Invariant Causal Mechanisms}, 
      author={Jovana Mitrovic and Brian McWilliams and Jacob Walker and Lars Buesing and Charles Blundell},
      year={2020},
      eprint={2010.07922},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training}, 
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
