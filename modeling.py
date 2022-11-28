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
            )[0]
        
        return context_output    # (batch size, sequence length, hidden state size)

    
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
            input_ids = input_ids[:,0,:],
            attention_mask= attention_mask[:,0,:],
            token_type_ids= token_type_ids[:,0,:],
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            )[0]       # (batch size, sequence length, hidden state size) 

        candidate_output = self.bert(
            input_ids = candidate_input_ids[:,0,:],
            attention_mask = candidate_attention_mask[:,0,:], 
            token_type_ids = candidate_token_type_ids[:,0,:],
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            )[0]       # (batch size, sequence length, hidden state size)

        dot_product = torch.matmul(context_output[:, 0, :], candidate_output[:, 0, :].t())    # (batch size, hidden state) @ (batch size, hidden state).t() = (batch size, batch size)

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

    
    def dot_attention_negatives(self, q, k, v):
        # q:   [bs, poly_m, hidden state] or [bs, 1, hidden state]
        # k=v: [bs, length, hidden state] or [bs, poly_m, hidden state]
        outputs = torch.empty(k.shape[0], k.shape[0], 1, k.shape[2]).to(q.device)
        for i in range(len(k)):
            attention_weights = torch.matmul(q, k[i].unsqueeze(0).permute(0, 2, 1))
            attention_weights = F.softmax(attention_weights, -1)

            outputs[i] = torch.matmul(attention_weights, v[i].unsqueeze(0))
            
        return outputs
    

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
            )[0]                        # (1, sequence length, hidden state)

        code_embedding_ids = torch.arange(self.m, dtype = torch.long).to(context_output.device)
        code_embedding_ids = code_embedding_ids.unsqueeze(0).expand(context_output.shape[0], self.m)

        codes = self.code_embedding(code_embedding_ids)                             # (1, m, hidden state)

        code_output = self.dot_attention(codes, context_output, context_output)     # (1, m, hidden state)

        cross_output = self.dot_attention(candidate_output.unsqueeze(1), code_output, code_output)      # (candidate size, 1, hidden state)

        assert codes.shape == torch.Size([1, code_embedding_ids.shape[1], 768])
        assert code_output.shape == torch.Size([1, code_embedding_ids.shape[1], 768])
        assert cross_output.shape == torch.Size([len(candidate_output), 1, 768])

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
        ):
        
        context_output = self.bert(
            input_ids = input_ids[:,0,:],
            attention_mask= attention_mask[:,0,:],
            token_type_ids= token_type_ids[:,0,:],
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
            input_ids = candidate_input_ids[:,0,:],
            attention_mask = candidate_attention_mask[:,0,:], 
            token_type_ids = candidate_token_type_ids[:,0,:],
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            )[0][:, 0, :]           # (batch size, hidden state)

        cross_outputs = self.dot_attention_negatives(candidate_output.unsqueeze(1), code_output, code_output)   # (batch size, batch size, 1, hidden state)

        dot_product = torch.empty(cross_outputs.shape[0], cross_outputs.shape[0]).to(cross_outputs.device)
        for i ,cross_output in enumerate(cross_outputs):
            dot_product[i] = torch.sum(cross_output.squeeze(1) * candidate_output, dim = -1)                       # (batch size)

        loss_fnt = nn.CrossEntropyLoss()
        loss = loss_fnt(dot_product, torch.arange(dot_product.shape[0]).to(dot_product.device))

        return loss, dot_product
        

class CrossEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 2)
        
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        labels = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        ):
        
        pooled_output = self.bert(
            input_ids = input_ids[:,0,:],
            attention_mask= attention_mask[:,0,:],
            token_type_ids= token_type_ids[:,0,:],
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            ).pooler_output       # (batch size, hidden state size)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)

        if labels is not None:
            labels = labels.to(logits.device)
            loss_fnt = nn.CrossEntropyLoss()
            loss = loss_fnt(logits, labels)

            return loss, None
        else:
            return None, logits

