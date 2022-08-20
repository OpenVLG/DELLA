import torch
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    latent: Optional[Tuple[torch.FloatTensor]] = None
    kl_loss: Optional[torch.FloatTensor] = None
    log_prior: Optional[torch.FloatTensor] = None
    log_post: Optional[torch.FloatTensor] = None
    dist_parameter: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class DecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    bow_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None