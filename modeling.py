import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BiEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        candidate_input_ids = None,
        candidate_attention_mask = None,
        candidate_token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        training = True
        ):
        
        context_output = self.bert(
                                input_ids = input_ids,
                                attention_mask= attention_mask,
                                token_type_ids= token_type_ids,
                                position_ids = position_ids,
                                head_mask = head_mask,
                                inputs_embeds = inputs_embeds,
                                output_attentions = output_attentions,
                                output_hidden_states = output_hidden_states,
                                )

        if training:
            candidate_output = self.bert(input_ids = candidate_input_ids,
                                    attention_mask = candidate_attention_mask, 
                                    token_type_ids = candidate_token_type_ids,
                                    position_ids = position_ids,
                                    head_mask = head_mask,
                                    inputs_embeds = inputs_embeds,
                                    output_attentions = output_attentions,
                                    output_hidden_states = output_hidden_states,
                                    )

            dot_product = torch.matmul(context_output[0][:, 0, :], candidate_output[0][:, 0, :].t())

            loss_fnt = nn.CrossEntropyLoss()
            loss = loss_fnt(dot_product, torch.arange(dot_product.shape[0]).to(dot_product.device))

            return loss, dot_product

        return context_output[1]
