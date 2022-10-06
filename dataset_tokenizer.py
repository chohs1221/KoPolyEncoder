class TokenizeDataset:
    def __init__(self, context, candidate, tokenizer, return_tensors = None, device = None):
        self.context = list(context)
        self.candidate = list(candidate)

        self.tokenizer = tokenizer

        self.return_tensors = return_tensors
        self.device = device

    def __getitem__(self, i):
        if self.device is not None:
            context_input = self.tokenizer(self.context[i], padding='max_length', max_length=360, truncation=True, return_tensors=self.return_tensors).to(self.device)
            candidate_input = self.tokenizer(self.candidate[i], padding='max_length', max_length=360, truncation=True, return_tensors=self.return_tensors).to(self.device)
        elif self.device is None:
            context_input = self.tokenizer(self.context[i], padding='max_length', max_length=360, truncation=True, return_tensors=self.return_tensors)
            candidate_input = self.tokenizer(self.candidate[i], padding='max_length', max_length=360, truncation=True, return_tensors=self.return_tensors)

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