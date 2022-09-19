import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel


class BiEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    

    def encode(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        ):

        context_output = self.bert(
            input_ids = input_ids,
            attention_mask= attention_mask,
            token_type_ids= token_type_ids,
            )
        
        return context_output[0]    # (batch size, sequence length, hidden state size)

    
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
            )       # (batch size, sequence length, hidden state size) 

        candidate_output = self.bert(
            input_ids = candidate_input_ids,
            attention_mask = candidate_attention_mask, 
            token_type_ids = candidate_token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            )       # (batch size, sequence length, hidden state size)

        dot_product = torch.matmul(context_output[0][:, 0, :], candidate_output[0][:, 0, :].t())    # (batch size, hidden state) @ (batch size, hidden state).t() = (batch size, batch size)

        loss_fnt = nn.CrossEntropyLoss()
        loss = loss_fnt(dot_product, torch.arange(dot_product.shape[0]).to(dot_product.device))

        return loss, dot_product


class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, m):
        super().__init__(config)

        self.bert = BertModel(config)

        self.m = m
        self.code_embedding = nn.Embedding(m, config.hidden_size)
        
        # Initialize weights and apply final processing
        self.post_init()

    
    def dot_attention(self, q, k, v):
        # q:   [bs, poly_m, hidden state] or [bs, 1, hidden state]
        # k=v: [bs, length, hidden state] or [bs, poly_m, hidden state]
        attention_weights = torch.matmul(q, k.permute(0, 2, 1))
        attention_weights = F.softmax(attention_weights, -1)

        output = torch.matmul(attention_weights, v)     # (batch size, m, hidden state) or (batch size, 1, hidden state)

        return output
    

    def encode(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        ):
        
        context_output = self.bert(
            input_ids = input_ids,
            attention_mask= attention_mask,
            token_type_ids= token_type_ids,
            )[0]

        return context_output           # (batch size, sequence length, hidden state)


    def context_encode(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        candidate_output = None,
        ):
        
        context_output = self.bert(
            input_ids = input_ids,
            attention_mask= attention_mask,
            token_type_ids= token_type_ids,
            )[0]                        # (batch size, sequence length, hidden state)

        code_embedding_ids = torch.arange(self.m, dtype = torch.long).to(context_output.device)
        code_embedding_ids = code_embedding_ids.unsqueeze(0).expand(context_output.shape[0], self.m)

        codes = self.code_embedding(code_embedding_ids)                             # (m, hidden state)

        code_output = self.dot_attention(codes, context_output, context_output)     # (batch size, m, hidden state)

        cross_output = self.dot_attention(candidate_output.unsqueeze(1), code_output, code_output)

        return cross_output


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
        candidate_output = None,
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
            )[0]                        # (batch size, sequence length, hidden state)

        code_embedding_ids = torch.arange(self.m, dtype = torch.long).to(context_output.device)
        code_embedding_ids = code_embedding_ids.unsqueeze(0).expand(context_output.shape[0], self.m)

        codes = self.code_embedding(code_embedding_ids)                             # (m, hidden state)

        code_output = self.dot_attention(codes, context_output, context_output)     # (batch size, m, hidden state)

        candidate_output = self.bert(
            input_ids = candidate_input_ids,
            attention_mask = candidate_attention_mask, 
            token_type_ids = candidate_token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            )[0][:, 0, :]           # (batch size, hidden state)

        cross_output = self.dot_attention(candidate_output.unsqueeze(1), code_output, code_output)  # (batch size, 1, hidden state)
        
        dot_product = torch.matmul(cross_output.squeeze(1), candidate_output.t())                   # (batch size, hidden state) @ (batch size, hidden state).t() = (batch size, batch size)

        loss_fnt = nn.CrossEntropyLoss()
        loss = loss_fnt(dot_product, torch.arange(dot_product.shape[0]).to(dot_product.device))

        return loss, dot_product
        