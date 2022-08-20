# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import os
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, RNNCell

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import GPT2Config, EncoderDecoderModel, EncoderDecoderConfig

from dist import Normal
from modeloutput import EncoderOutput, DecoderOutput
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]

class LMFLayer(nn.Module):
    def __init__(self, config):
        super(LMFLayer, self).__init__()
        self.rank = config.latent_lmf_rank
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size
        
        self.text_factor = nn.ModuleList([nn.Linear(self.hidden_size + 1, self.hidden_size) for _ in range(self.rank)])
        self.latent_factor = nn.ModuleList([nn.Linear(self.latent_size + 1, self.hidden_size) for _ in range(self.rank)])

    def forward(self, hidden_states, latent):
        '''
        Args:
            hidden_states: tensor of shape (batch_size, sequence_len, text_in)
            latent: tensor of shape (batch_size, latent_size)
        '''
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        text_cat = torch.ones(batch_size, seq_len, 1, dtype=torch.float, device=device)
        latent_cat = torch.ones(batch_size, 1, dtype=torch.float, device=device)
        hidden_states = torch.cat((hidden_states, text_cat), dim=-1)
        latent = torch.cat((latent, latent_cat), dim=-1)
        text_fusion_output = []
        latent_fusion_output = []
        for text_factor, latent_factor in zip(self.text_factor, self.latent_factor):
            text_fusion = text_factor(hidden_states)
            latent_fusion = latent_factor(latent)
            text_fusion_output.append(text_fusion)
            latent_fusion_output.append(latent_fusion)
        text_fusion = torch.stack(text_fusion_output).sum(0)
        latent_fusion = torch.stack(latent_fusion_output).sum(0)
        text_fusion = text_fusion.transpose(1, 0)
        output = text_fusion * latent_fusion
        output = output.transpose(1, 0)
        return output

class AverageSelfAttention(nn.Module):
    def __init__(self, config):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(config.hidden_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = ACT2FN[config.activation_function]

    def forward(self, inputs, attention_mask=None):
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        if attention_mask is not None:
            scores = scores + attention_mask

        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze(1)

        return representations, scores

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.lmf = LMFLayer(config)
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        self.config = config

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, use_causal_mask=True):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if use_causal_mask:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        use_causal_mask=False,
        latent=None,
    ):
        if latent is not None:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            value = self.lmf(value, latent)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, use_causal_mask=use_causal_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        latent=None,
        use_cache=False,
        output_attentions=False,
        use_causal_mask=False,
        **latent_inject_kwargs,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            use_causal_mask=use_causal_mask,
            latent=latent,
            **latent_inject_kwargs,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

GPT2_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0][0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.

            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.

            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48

    Example::

            # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
            device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8],

                          1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                          2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                          3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example::

        # On a 4 GPU machine with gpt2-large:
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7],

                    1: [8, 9, 10, 11, 12, 13, 14, 15],
                    2: [16, 17, 18, 19, 20, 21, 22, 23],
                    3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""

class GPT2VAEEncoder(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, wte, wpe, h, ln_f, dropout):
        super().__init__(config)

        self.wte = wte
        self.wpe = wpe
        self.drop = dropout
        self.h = h
        self.ln_f = ln_f

        self.num_layers = config.num_hidden_layers
        self.latent_size = config.latent_size
        self.kl_threshold = config.kl_threshold
        self.is_cvae = config.is_cvae

        self.begin_layer = config.begin_layer if config.begin_layer else 0
        self.end_layer = config.end_layer if config.end_layer else len(self.h) - 1
        assert 0 <= self.begin_layer <= self.end_layer <= len(self.h) - 1

        self.recog_network = nn.ModuleList([nn.Linear(config.hidden_size + self.latent_size, 2 * self.latent_size) for _ in range(self.num_layers)])
        if self.is_cvae:
            self.prior_network = nn.ModuleList([nn.Linear(self.latent_size + config.hidden_size, 2 * self.latent_size) for _ in range(self.num_layers)])
        else:
            self.prior_network = nn.ModuleList([nn.Linear(self.latent_size, 2 * self.latent_size) for _ in range(self.num_layers)])
        self.reccurnt_cell_weight_hh = nn.ModuleList([nn.Linear(self.latent_size, self.latent_size) for _ in range(self.num_layers)])
        self.reccurnt_cell_weight_ih = nn.ModuleList([nn.Linear(self.latent_size, self.latent_size) for _ in range(self.num_layers)])
        self.pooling = nn.ModuleList([AverageSelfAttention(config) for _ in range(len(h))])
        self.tanh_activate = nn.Tanh()

        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
                self.pooling[block] = self.pooling[block].to(cuda_device)
                self.recog_network[block] = self.recog_network[block].to(cuda_device)
                self.prior_network[block] = self.prior_network[block].to(cuda_device)
                self.reccurnt_cell_weight_hh[block] = self.reccurnt_cell_weight_hh[block].to(cuda_device)
                self.reccurnt_cell_weight_ih[block] = self.reccurnt_cell_weight_ih[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
            self.pooling[index] = self.pooling[index].to("cpu")
            self.recog_network[index] = self.recog_network[index].to("cpu")
            self.prior_network[index] = self.prior_network[index].to("cpu")
            self.reccurnt_cell_weight_hh[index] = self.reccurnt_cell_weight_hh[index].to("cpu")
            self.reccurnt_cell_weight_ih[index] = self.reccurnt_cell_weight_ih[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def _recurrent_latent(self, post_latent, prior_latent_representaion, weight_ih, weight_hh):
        h = self.tanh_activate(weight_hh(prior_latent_representaion) + weight_ih(post_latent))
        return h

    def get_prior(self, batch_size, device, condition=None, condition_mask=None):
        if condition is not None:
            condition_output = self(condition, attention_mask=condition_mask, compute_kl=False)
            condition_hidden_states = condition_output.hidden_states
        else:
            condition_hidden_states = tuple([None] * self.num_layers)
        prior_dist = Normal.get_standard(batch_size, self.latent_size, device)
        prior_latent_representaion = torch.zeros(batch_size, self.latent_size).to(device)
        prior_latent = torch.zeros_like(prior_latent_representaion)
        all_prior_latent = ()

        for idx in range(self.begin_layer, self.end_layer + 1):
            cond = condition_hidden_states[idx - self.begin_layer]
            prior_network = self.prior_network[idx - self.begin_layer]
            weight_ih = self.reccurnt_cell_weight_ih[idx - self.begin_layer]
            weight_hh = self.reccurnt_cell_weight_hh[idx - self.begin_layer]
            prior_latent_representaion = self._recurrent_latent(prior_latent, prior_latent_representaion, weight_ih, weight_hh)
            if cond is not None:
                prior_representaion = torch.cat((prior_latent_representaion, cond), dim=-1)
            else:
                prior_representaion = prior_latent_representaion
            prior_mu, prior_sigma = torch.chunk(prior_network(prior_representaion), 2, dim=-1)
            prior_dist = Normal(prior_mu, prior_sigma)
            prior_latent, _ = prior_dist.sample()
            all_prior_latent = all_prior_latent + (prior_latent, )
        return all_prior_latent

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        return_dict=None,
        condition_hidden_states=None,
        compute_kl=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        if condition_hidden_states is None:
            condition_hidden_states = tuple([None] * len(self.h))
        else:
            assert self.is_cvae
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        prior_dist = Normal.get_standard(batch_size, self.latent_size, device)
        prior_latent_representaion = torch.zeros(batch_size, self.latent_size).to(device)
        post_latent = torch.zeros_like(prior_latent_representaion)
        all_post_latent = ()
        all_kl_loss = ()
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = ()
        all_log_prior = ()
        all_log_post = ()
        all_post_mu = ()
        all_post_sigma = ()
        loops = zip(self.h, past_key_values, self.pooling, condition_hidden_states, self.recog_network, self.prior_network, self.reccurnt_cell_weight_ih, self.reccurnt_cell_weight_hh)
        for i, (block, layer_past, pooling_layer, condition, recog_network, prior_network, weight_ih, weight_hh) in enumerate(loops):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                use_causal_mask=False,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            representation, _ = pooling_layer(hidden_states)
            all_hidden_states = all_hidden_states + (representation, )
            if compute_kl and self.begin_layer <= i <= self.end_layer:
                # update prior latent
                prior_latent_representaion = self._recurrent_latent(post_latent, prior_latent_representaion, weight_ih, weight_hh)
                if condition is not None:
                    prior_representaion = torch.cat((prior_latent_representaion, condition), dim=-1)
                else:
                    prior_representaion = prior_latent_representaion
                prior_mu, prior_sigma = torch.chunk(prior_network(prior_representaion), 2, dim=-1)
                prior_dist = Normal(prior_mu, prior_sigma)

                post_latent_representation = recog_network(torch.cat((representation, prior_latent_representaion), dim=-1))
                post_mu, post_sigma = torch.chunk(post_latent_representation, 2, dim=-1)
                post_dist = Normal(post_mu, post_sigma)
                post_latent, _ = post_dist.sample()
                all_post_latent = all_post_latent + (post_latent, )
                kl_loss = post_dist.kl(prior_dist)
                kl_threshold = torch.Tensor([self.kl_threshold]).type_as(kl_loss)
                kl_loss = torch.max(kl_loss, kl_threshold)

                log_prior_z = prior_dist.log_p(post_latent)
                log_post_z = post_dist.log_p(post_latent)
                if self.model_parallel:
                    kl_loss = kl_loss.to(self.first_device)
                    log_prior_z = log_prior_z.to(self.first_device)
                    log_post_z = log_post_z.to(self.first_device)
                all_kl_loss = all_kl_loss + (kl_loss, )
                all_log_prior = all_log_prior + (log_prior_z, )
                all_log_post = all_log_post + (log_post_z, )
                all_post_mu = all_post_mu + (post_mu, )
                all_post_sigma = all_post_sigma + (post_sigma, )
                

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        
        if compute_kl:
            kl_loss = torch.stack(all_kl_loss).mean(0)
            log_prior = torch.stack(all_log_prior).mean(0)
            log_post = torch.stack(all_log_post).mean(0)

            mu = torch.cat(all_post_mu, dim=-1)
            sigma = torch.cat(all_post_sigma, dim=-1)
        else:
            kl_loss = None
            log_prior = None
            log_post = None
            mu, sigma = None, None

        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            hidden_states = hidden_states.to(self.first_device)

        return EncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            latent=all_post_latent,
            dist_parameter=(mu, sigma),
            kl_loss=kl_loss,
            log_prior=log_prior,
            log_post=log_post,
        )

class GPT2VAEDecoderWithLMHead(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, wte, wpe, h, ln_f, dropout):
        super().__init__(config)

        self.wte = wte
        self.wpe = wpe
        self.drop = dropout
        self.h = h
        self.ln_f = ln_f

        self.init_weights()
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.begin_layer = config.begin_layer if config.begin_layer else 0
        self.end_layer = config.end_layer if config.end_layer else len(self.h) - 1
        assert 0 <= self.begin_layer <= self.end_layer <= len(self.h) - 1

        self.use_bow = config.use_bow
        if self.use_bow:
            self.mlp_bow = nn.Linear(config.latent_size, config.n_embd)
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.lm_head = self.lm_head.to(self.first_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.lm_head = self.lm_head.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        latent = kwargs.get("latent", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "all_latent": latent,
        }

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        all_latent=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        all_hidden_states = ()
        presents = ()
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            
            if self.begin_layer <= i <= self.end_layer:
                latent = all_latent[i - self.begin_layer]
            else:
                latent = None
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
                if latent is not None:
                    latent = latent.to(hidden_states.device)
            
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                latent=latent,
                use_cache=use_cache,
                output_attentions=output_attentions,
                use_causal_mask=True,
            )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        lm_logits = self.lm_head(hidden_states)
        if self.use_bow:
            bow_logits = self.lm_head(self.mlp_bow(all_latent[-1]))
        else:
            bow_logits = None
        return DecoderOutput(
            logits=lm_logits,
            bow_logits=bow_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

class Della(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        wte = nn.Embedding(config.vocab_size, config.hidden_size)
        wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        drop = nn.Dropout(config.embd_pdrop)

        self.encoder = GPT2VAEEncoder(config, wte, wpe, h, ln_f, drop)
        self.decoder = GPT2VAEDecoderWithLMHead(config, wte, wpe, h, ln_f, drop)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, reduce=False)
        self.loss_fn_reduced = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.h))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.first_device = self.encoder.first_device
        self.model_parallel = True
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        torch.cuda.empty_cache()

    def get_encode_states(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mu, sigma = encoder_outputs.dist_parameter
        return mu, sigma

    def get_reduced_celoss(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        latent = encoder_outputs.latent
        decoder_outputs = self.decoder(input_ids, attention_mask=attention_mask, all_latent=latent)
        lm_logits = decoder_outputs.logits
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
    
        ce_loss = self.loss_fn_reduced(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return ce_loss

    def get_celoss(self, input_ids, attention_mask, latent):
        batch_size = input_ids.size(0)
        decoder_outputs = self.decoder(input_ids, attention_mask=attention_mask, all_latent=latent)
        lm_logits = decoder_outputs.logits
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
    
        ce_loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ce_loss = ce_loss.view(batch_size, -1).sum(-1)
        return ce_loss

    def get_klloss(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return encoder_outputs.kl_loss

    def get_neg_entropy(self, input_ids, attention_mask, ns=30):
        log_post_list = []
        for _ in range(ns):
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            log_post_list.append(encoder_outputs.log_post)
        neg_entropy = torch.stack(log_post_list).mean(0).sum()
        latentoutput = encoder_outputs.latent
        latent = []
        for l in latentoutput:
            if self.model_parallel:
                temp = l.to(self.first_device)
                latent.append(temp)
            else:
                latent.append(l)
        latent = torch.cat(latent, dim=-1)
        return neg_entropy, latent

    def iw_sample(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        latent = encoder_outputs.latent
        log_prior = encoder_outputs.log_prior
        log_post = encoder_outputs.log_post
        log_gen = -self.get_celoss(input_ids=input_ids, attention_mask=attention_mask, latent=latent)
        log_likelihood = log_gen + log_prior - log_post
        return log_gen, log_likelihood

    def get_prior(self, batch_size, device, condition=None, condition_mask=None):
        return self.encoder.get_prior(batch_size, device, condition=condition, condition_mask=condition_mask)

    def forward(self, input_ids, labels=None, attention_mask=None, condition=None, condition_mask=None):
        if condition is not None:
            condition_output = self.encoder(condition, attention_mask=condition_mask, compute_kl=False)
            condition = condition_output.hidden_states
        else: 
            condition = None

        encoder_output = self.encoder(input_ids, attention_mask=attention_mask, condition_hidden_states=condition)
        batch_size = input_ids.size(0)
        latent = encoder_output.latent
        kl_loss = encoder_output.kl_loss.mean()
        decoder_output = self.decoder(input_ids, attention_mask=attention_mask, all_latent=latent)
        lm_logits = decoder_output.logits
        bow_logits = decoder_output.bow_logits

        shift_logits = lm_logits[..., :-1, :].contiguous()
        if labels is None:
            shift_labels = input_ids[..., 1:].contiguous()
        else:
            shift_labels = labels[..., 1:].contiguous()
    
        ce_loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ce_loss = ce_loss.view(batch_size, -1).sum(-1)
        ce_loss = ce_loss.mean()

        seq_len = input_ids.size(-1)
        if bow_logits is not None:
            bow_logits = bow_logits.unsqueeze(1).repeat(1, seq_len, 1)[..., :-1, :].contiguous()
            bow_loss = self.loss_fn(bow_logits.view(-1, bow_logits.size(-1)), shift_labels.view(-1))
            bow_loss = bow_loss.view(batch_size, -1).sum(-1)
            bow_loss = bow_loss.mean()
            return ce_loss, kl_loss, bow_loss, encoder_output, decoder_output
        else:
            return ce_loss, kl_loss, encoder_output, decoder_output
