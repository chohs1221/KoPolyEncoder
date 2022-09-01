class DataLoader:
    def __init__(self, context, candidate, tokenizer):
        self.context = context
        self.candidate = candidate

        self.tokenizer = tokenizer

    def __getitem__(self, i):
        context_input = self.tokenizer(self.context[i], padding='max_length', max_length=50, truncation=True)
        candidate_input = self.tokenizer(self.candidate[i], padding='max_length', max_length=50, truncation=True)

        return {
            'input_ids': context_input['input_ids'],
            'attention_mask': context_input['attention_mask'],
            'token_type_ids': context_input['token_type_ids'],
            'candidate_input_ids': candidate_input['input_ids'],
            'candidate_attention_mask': candidate_input['attention_mask'],
            'candidate_token_type_ids': candidate_input['token_type_ids']
                }

    def __len__(self):
        return len(self.context)