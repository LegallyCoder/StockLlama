from torch import nn
from torch.nn import functional as F
import torch

from configuration_stockllama import StockLlamaConfig

from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
from typing import Any, Dict, List, Optional, Tuple , Union

class FloatEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx ,term_number):
        super(FloatEmbedding, self).__init__()
        self.term_number = term_number
        self.int_part = nn.Embedding(vocab_size, hidden_size ,padding_idx)
        self.float_part = nn.Embedding(10**term_number , hidden_size)
        
    def forward(self, input):
        float_input = ((input - torch.floor(input)) * (10**self.term_number)).to(torch.long)
        int_input = input.to(torch.long)
        output = self.float_part(float_input) + self.int_part(int_input)

        return output
    
class StockLlamaPreTrainedModel(LlamaPreTrainedModel):
    config_class = StockLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class StockLlamaModel(LlamaModel):
    config_class = StockLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self._use_flash_attention_2 = True
        self.embed_tokens = FloatEmbedding(config.vocab_size, config.hidden_size, self.padding_idx, config.term_number)
        self.post_init()


class StockLlamaForForecasting(StockLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = StockLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)  
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = MSELoss()
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )