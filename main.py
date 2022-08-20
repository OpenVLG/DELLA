import argparse
import logging
import os
import json
import torch
import random
import numpy as np
import time

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from dataset import VAEDataset, WPDataset
from train import train, valid, generate

from model import Della

from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='./data/yelp/yelp.train.txt', type=str,
                        help="Data path for training.")
    parser.add_argument("--valid_file", default='./data/yelp/yelp.train.txt', type=str,
                        help="Data path for valid")
    parser.add_argument("--test_file", default='./data/yelp/yelp.train.txt', type=str,
                        help="Data path for test")
    parser.add_argument("--pretrained_model", type=str, default='gpt2', 
                        help="Pretrained model to be loaded")
    parser.add_argument("--dataset_type", type=str, default='vae', choices=['vae', 'wp'], 
                        help="Dataset type")
    parser.add_argument("--output_dir", default='./checkpoints', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_name", default='della', type=str,
                        help="The model name")
    parser.add_argument("--generation_output_dir", default='./generation_output', type=str,
                        help="The output directory where the log will be written.")
    # Other parameters\
    parser.add_argument("--load_epoch", default=None, type=int, help="the epochs of trained model to load")
    parser.add_argument("--epochs", default=40, type=int, help="total epochs")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,help="Batch size per GPU for training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--kl_threshold", default=0, type=float,
                        help="The threshold of the minimum KL value, default as 0")
    parser.add_argument("--latent_size", default=32, type=int,
                        help="The dimension of latent space")
    parser.add_argument("--latent_lmf_rank", default=4, type=int,
                        help="latent size")
    parser.add_argument("--max_length", default=200, type=int,
                        help="Max length for generation")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument('--log_step', type=int, default=100,
                        help="Steps for logging")
    parser.add_argument('--num_beams', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--greedy_decoding', action='store_true',
                        help="Choose to use greedy decoding")
    parser.add_argument('--top_k', type=int, default=-1, help='Set top k')
    parser.add_argument('--top_p', type=float, default=0.9, help='Set top p')
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--model_parallel', action='store_true', 
                        help="Choose to use model parallel, mapping the layers to different devices")
    parser.add_argument('--eval', action='store_true', help='Choose to eval the model')
    parser.add_argument('--eval_metrics', action='store_true',
                        help="Choose to eval the metrics for representation learning")
    parser.add_argument('--generation', action='store_true', help='Choose to generate')
    parser.add_argument('--use_scheduler', action='store_true',
                        help="Choose to use lr scheduler")
    parser.add_argument('--cycle_annealing', action='store_true',
                        help="Choose to use cycle annealing")
    parser.add_argument('--cycle_iters', type=int, default=2,
                        help="Set the iters for cycle annealing")
    parser.add_argument('--sample_times', type=int, default=30,
                        help="The total times of sample when computing PPL with importance weighted sampling")
    parser.add_argument('--use_bow', action='store_true',
                        help="Choose to use bow loss")
    parser.add_argument('--bow_weight',type=float, default=0.2,
                        help="Set the weight of bow loss term")
    parser.add_argument("--begin_layer", default=None, type=int,
                        help="The beginning layer to consider the latent vector, default as the first layer of model")
    parser.add_argument("--end_layer", default=None, type=int,
                        help="The end layer to consider the latent vector, default as the last layer of model")
    args = parser.parse_args()
    return args

def prepare(args):
    torch.set_num_threads(3)

    if not args.eval and not args.generation:
        os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(
            args.output_dir, args.model_name, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    if args.no_cuda:
        args.n_gpu = 1
    else:
        args.n_gpu = torch.cuda.device_count()
    args.batch_size = args.per_gpu_train_batch_size * args.n_gpu
    
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    if args.no_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0')

def init_para_frompretrained(model, gpt2):
    logger.info('load gpt2 pretrained model parameters')
    model = model.encoder
    model.wte.weight = gpt2.wte.weight
    model.wpe.weight = gpt2.wpe.weight

    for i in range(len(gpt2.h)):
        model.h[i].ln_1.weight = gpt2.h[i].ln_1.weight
        model.h[i].ln_1.bias = gpt2.h[i].ln_1.bias
        model.h[i].attn.c_attn.weight = gpt2.h[i].attn.c_attn.weight
        model.h[i].attn.c_attn.bias = gpt2.h[i].attn.c_attn.bias
        model.h[i].attn.c_proj.weight = gpt2.h[i].attn.c_proj.weight
        model.h[i].attn.c_proj.bias = gpt2.h[i].attn.c_proj.bias
        model.h[i].ln_2.weight = gpt2.h[i].ln_2.weight
        model.h[i].ln_2.bias = gpt2.h[i].ln_2.bias
        model.h[i].mlp.c_fc.weight = gpt2.h[i].mlp.c_fc.weight
        model.h[i].mlp.c_fc.bias = gpt2.h[i].mlp.c_fc.bias
        model.h[i].mlp.c_proj.weight = gpt2.h[i].mlp.c_proj.weight
        model.h[i].mlp.c_proj.bias = gpt2.h[i].mlp.c_proj.bias

    model.ln_f.weight = gpt2.ln_f.weight
    model.ln_f.bias = gpt2.ln_f.bias

def prepare_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if '<s>' not in tokenizer.vocab:
        tokenizer._add_tokens(['<s>'])
    if '</s>' not in tokenizer.vocab:
        tokenizer._add_tokens(['</s>'])
    tokenizer.pad_id = 50256
    
    tokenizer.bos_id = tokenizer.convert_tokens_to_ids('<s>')
    tokenizer.eos_id = tokenizer.convert_tokens_to_ids('</s>')

    model_config = AutoConfig.from_pretrained(args.pretrained_model)
    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_id
    model_config.kl_threshold = args.kl_threshold
    model_config.is_cvae = (args.dataset_type == 'wp')
    model_config.use_bow = args.use_bow
    model_config.begin_layer = args.begin_layer
    model_config.end_layer = args.end_layer

    for arg in vars(args):
        if arg.startswith('latent'):
            setattr(model_config, arg, getattr(args, arg))
    
    model = Della(model_config)
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model)
    logging.info('loading pretrained model parameters...')
    init_para_frompretrained(model, pretrained_model)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.decoder.wte = model.encoder.wte
    if args.load_epoch is not None:
        model_path = os.path.join(args.output_dir, args.model_name, 'model_epoch_{}.pt'.format(args.load_epoch))
        model_state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(model_state_dict)
        logging.info('load model_epoch_{}.pt finish'.format(args.load_epoch))
    else:
        args.load_epoch = -1

    if args.model_parallel and torch.cuda.device_count() > 1:  
        logging.info('model paralleize...')
        model.parallelize()
    else:
        model = model.to(args.device)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
    return model, tokenizer

def prepare_data(tokenizer, args):
    dataset_class = {'vae': VAEDataset, 'wp': WPDataset}
    if args.eval or args.generation:
        logging.info("eval model: the epoch {} of {}".format(args.load_epoch, args.model_name))
        test_dataset = dataset_class[args.dataset_type](args.test_file, tokenizer, args.device)
        test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        return test_iter
    else:
        train_dataset = dataset_class[args.dataset_type](args.train_file, tokenizer, args.device)
        valid_dataset = dataset_class[args.dataset_type](args.valid_file, tokenizer, args.device)
        train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_iter = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)
        logging.info('training with {} samples...'.format(len(train_dataset)))
        return train_iter, valid_iter

def main():
    args = get_args()
    prepare(args)
    model, tokenizer = prepare_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info('total parameters: {}'.format(total_params))
    if args.eval or args.generation:
        test_iter = prepare_data(tokenizer, args)
        if args.eval:
            valid(model, test_iter, args.load_epoch, args)
        if args.generation:
            generate(model, test_iter, tokenizer, args)
    else:
        train_iter, valid_iter = prepare_data(tokenizer, args)
        train(model, train_iter, valid_iter, args)

if __name__ == "__main__":
    main()
