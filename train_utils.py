import numpy as np
from dist import Normal
from tqdm import tqdm
import torch
import math

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

def calc_iwnll(model, iters, ns=30):
    report_kl_loss = report_ce_loss = report_loss = 0
    report_num_words = report_num_sents = 0
    for inputs in tqdm(iters, desc="Evaluating PPL"):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        seq_len = attention_mask.sum(-1).long()
        report_num_sents += input_ids.size(0)
        report_num_words += seq_len.sum().item()
        kl_loss = model.get_klloss(input_ids, attention_mask)
        ll_tmp = []
        ce_tmp = []
        for _ in range(ns):
            log_gen, log_likelihood = model.iw_sample(input_ids, attention_mask)
            ce_tmp.append(log_gen)
            ll_tmp.append(log_likelihood)

        ll_tmp = torch.stack(ll_tmp, dim=0)
        log_prob_iw = log_sum_exp(ll_tmp, dim=0) - math.log(ns)
        log_ce_iw = torch.mean(torch.stack(ce_tmp), dim=0)
        report_kl_loss += kl_loss.sum().item()
        report_ce_loss += log_ce_iw.sum().item()
        report_loss += log_prob_iw.sum().item()

    elbo = (report_kl_loss - report_ce_loss) / report_num_sents
    nll = - report_ce_loss / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(-report_loss / report_num_words)
    return ppl, elbo, nll, kl

def calc_au(model, iters, delta=0.2):
    """compute the number of active units
    """
    cnt = 0
    for inputs in tqdm(iters, desc="Evaluating Active Units, Stage 1"):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        mean, _ = model.get_encode_states(input_ids=input_ids, attention_mask=attention_mask)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for inputs in tqdm(iters, desc="Evaluating Active Units, Stage 2"):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        mean, _ = model.get_encode_states(input_ids=input_ids, attention_mask=attention_mask)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)
    au = (au_var >= delta).sum().item()
    au_prop = au / mean.size(-1)
    return au_prop