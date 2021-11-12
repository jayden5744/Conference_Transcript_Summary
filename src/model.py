
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from abc import ABC

import pytorch_lightning as pl
from transformers import AdamW, EncoderDecoderModel, get_linear_schedule_with_warmup, BertConfig, EncoderDecoderModel, \
    BertLMHeadModel, add_start_docstrings, EncoderDecoderConfig
from transformers.utils import logging
from transformers.file_utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertPreTrainedModel, BertModel, \
    BertOnlyMLMHead, BERT_INPUTS_DOCSTRING

from src.loss_function import label_smoothed_nll_loss

logger = logging.get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_CONFIG_FOR_DOC = "BertConfig"


@add_start_docstrings(
    """ BERT Model with a `language modeling` head on top for CLM fine-tuning.""", BERT_START_DOCSTRING
)
class BertLMHeadModelIR(BertPreTrainedModel, ABC):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super(BertLMHeadModelIR, self).__init__(config)
        if not config.is_decoder:
            config.is_decoder = True
        self.vocab_size = config.vocab_size
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.exrl_cls = nn.Linear(768, 2)

        self.init_weights()

    def get_output_embeddings(self) -> nn.Module:
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            exrl_labels=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> config.is_decoder = True
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.training:
            sequence_output = F.dropout(sequence_output, p=0.2)
        prediction_scores = self.cls(sequence_output)

        if exrl_labels is not None:
            exrl_scores = self.exrl_cls(encoder_hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].log_softmax(dim=-1).contiguous()
            labels = labels[:, 1:].contiguous()
            # loss_fct = CrossEntropyLoss(ignore_index=0)
            lm_loss = label_smoothed_nll_loss(shifted_prediction_scores.view(-1, self.config.vocab_size),
                                              labels.view(-1, 1), 0.1)
            if exrl_labels is not None:
                exrl_loss_fct = CrossEntropyLoss(ignore_index=0)
                exrl_loss = exrl_loss_fct(exrl_scores.contiguous().view(-1, 2), exrl_labels.contiguous().view(-1))
                lm_loss = lm_loss + exrl_loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class Bert2Bert(ABC):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg.model_name
        self.model_cfg = cfg

        self.config = self.get_config()  # config
        self.model = self.get_model()   # model

    def forward(self, batch):
        return self.model(**batch)

    def get_config(self):
        config_encoder = BertConfig.from_pretrained(self.model_name)
        config_decoder = BertConfig.from_pretrained(self.model_name)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=config_encoder,
                                                                   decoder_config=config_decoder)
        config.tie_encoder_decoder = True
        config.encoder.hidden_dropout_prob = self.model_cfg.dp
        config.decoder.hidden_dropout_prob = self.model_cfg.dp
        return config

    def get_model(self):
        encoder = BertModel.from_pretrained(self.model_name)
        decoder = BertLMHeadModelIR(config=self.config.decoder)
        pt_decoder = BertLMHeadModel.from_pretrained(self.model_name)
        decoder.load_state_dict(pt_decoder.state_dict(), strict=False)
        model = EncoderDecoderModel(config=self.config, encoder=encoder, decoder=decoder)
        if self.model_cfg.pretrain_model:
            map_location = torch.device("cpu")
            state_dict = torch.load(
             self.model_cfg.pretrain_model_path,
             map_location=map_location
            )
            model.load_state_dict(state_dict, strict=False)

        model = model.cuda()
        return model

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        return self.model.zero_grad()

    def state_dict(self):
        return self.model.state_dict()

