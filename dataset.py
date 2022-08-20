from torch.utils.data import Dataset
import torch
import tqdm
import json
import logging
logger = logging.getLogger(__name__)

class VAEDataset(Dataset):
    def __init__(self, source_path, tokenizer, device=torch.device('cuda:0')):
        self.data = []
        self.tokenizer = tokenizer
        self.device = device
        with open(source_path) as f:
            for line in tqdm.tqdm(f, desc='Loading data...'):
                line = line.strip()
                if line == '':
                    continue
                line = line.split('\t')[-1]
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.data[idx])

    @staticmethod
    def create_mask(num_tokens, max_len):
        base_position_matrix = torch.arange(
            0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
        mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
        return mask

    def collate_fn(self, samples):
        samples = [[self.tokenizer.bos_id] + s + [self.tokenizer.eos_id] for s in samples]
        length_list = [len(s) for s in samples]
        max_t = max(length_list)
        new_samples = [s + [self.tokenizer.pad_id] * (max_t - len(s)) for s in samples]
        new_samples = torch.LongTensor(new_samples)
        attention_mask = self.create_mask(torch.LongTensor(length_list), max_t)
        return {
            'input_ids': new_samples.to(self.device),
            'attention_mask': attention_mask.byte().to(self.device),
        }

class WPDataset(Dataset):
    def __init__(self, source_path, tokenizer, device=torch.device('cuda:0'), max_length=700, add_prefix=False, add_special_token=False):
        self.source = []
        self.target = []
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.add_special_token = add_special_token
        self.add_prefix = add_prefix
        with open(source_path) as f:
            for line in tqdm.tqdm(f, desc='Loading data...'):
                line = json.loads(line.strip())
                source = line['source'].replace('<newline>', '\n')
                target = line['target'].replace('<newline>', '\n')
                if len(source.split()) + len(target.split()) < self.max_length:
                    self.source.append(source)
                    self.target.append(target)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source = self.tokenizer.encode(self.source[idx])
        target = self.tokenizer.encode(self.target[idx])
        return source, target

    @staticmethod
    def create_mask(num_tokens, max_len):
        base_position_matrix = torch.arange(
            0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
        mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
        return mask

    def collate_fn(self, samples):
        source_initial = [item[0] for item in samples]
        target_initial = [item[1] for item in samples]
        source = [s + [self.tokenizer.eos_id] for s in source_initial]
        target = [s + t + [self.tokenizer.eos_id] for s, t in zip(source, target_initial)]
        labels = [[self.tokenizer.pad_id] * len(s) + t + [self.tokenizer.eos_id] for s, t in zip(source, target_initial)]
        source = [item[:self.max_length] for item in source]
        target = [item[:self.max_length] for item in target]
        labels = [item[:self.max_length] for item in labels]

        source_length_list = [len(s) for s in source]
        source_max_t = max(source_length_list)
        new_source = [s + [self.tokenizer.pad_id] * (source_max_t - len(s)) for s in source]
        new_source = torch.LongTensor(new_source)
        source_attention_mask = self.create_mask(torch.LongTensor(source_length_list), source_max_t)

        target_length_list = [len(s) for s in target]
        target_max_t = max(target_length_list)
        new_target = [s + [self.tokenizer.pad_id] * (target_max_t - len(s)) for s in target]
        new_labels = [s + [self.tokenizer.pad_id] * (target_max_t - len(s)) for s in labels]
        new_target = torch.LongTensor(new_target)
        new_labels = torch.LongTensor(new_labels)
        target_attention_mask = self.create_mask(torch.LongTensor(target_length_list), target_max_t)

        return {
            'input_ids': new_target.to(self.device),
            'attention_mask': target_attention_mask.byte().to(self.device),
            'labels': new_labels.to(self.device),
            'condition': new_source.to(self.device),
            'condition_mask': source_attention_mask.byte().to(self.device),
        }