# DELLA

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official code for the paper [Fuse It More Deeply! A Variational Transformer with Layer-Wise Latent Variable Inference for Text Generation](https://arxiv.org/pdf/2207.06130.pdf) by Jinyi Hu, Xiaoyuan Yi, Wenhao Li, Maosong Sun, Xing Xie.

In this paper, we propose a novel variational Transformer framework, DELLA, to ameliorate the KL-vanishing problem and enhance the learning capacity of Transformer-based VAE. DELLA learns a series of layer-wise latent variables with each inferred from those of lower layers and tightly coupled with the hidden states by low-rank tensor product, achieving higher non-zero KL values even without any annealing or thresholding tricks. DELLA can improve both the quality and diversity of generated text.

### Usage

#### Prepare the dataset
We provide yelp and yahoo dataset in the Directory [data](https://github.com/OpenVLG/DELLA/tree/main/data). For unconditional generation task, just prepare the train, valid and test set where each line represents one training instance and input the dataset path when run the codes. For conditional generation task like story generation, prepare the dataset in the format of jsonl. Each line is a json like:
```
{'source': The prefix of the story, 'target': The main body of the story}
```

#### Training
For unconditional generation, run the codes with:
```
python main.py --train_file [path to training set] --valid_file [path to valid set] --per_gpu_train_batch_size 16 --model_name [config info of this training] --cycle_annealing
```

For conditional generation, run the codes with:
```
python main.py --train_file [path to training set] --valid_file [path to valid set] --dataset_type wp --per_gpu_train_batch_size 16 --model_name [config info of this training] --cycle_annealing
```

#### Generation
DELLA is available for all kinds of decoding strategy. For beam search (the number of beams is default as 10), run:
```
python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --num_beams 10
``` 
For greedy decoding, run:
```
python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --greedy_decoding
``` 
For top-k, top-p sampling, run:
```
python main.py --generation --test_file [path to test set] --model_name [config info of training] --load_epoch [the number of epoch to load] --top_k 50 --top_p 0.9
```

### Citation 
If you find this repo useful for your further research, please consider citing: 
```
@inproceedings{hu2022fuse,
  title={Fuse It More Deeply! A Variational Transformer with Layer-Wise Latent Variable Inference for Text Generation},
  author={Hu, Jinyi and Yi, Xiaoyuan and Li, Wenhao and Sun, Maosong and Xie, Xing},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={697--716},
  year={2022}
}
```
