from kobert_tokenizer import KoBERTTokenizer
import torch

from modeling import BiEncoder
from parsing import pickling

valid_context, valid_candidate = pickling('./data/pickle/valid.pickle', 'load')

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BiEncoder.from_pretrained('skt/kobert-base-v1')
model.to('cuda');

# 메모리 해결
candidate_text = ['내 이름은 달이라고 해',
                            '나는 145센치야',
                            "나는 그런거 몰라",
                            "나는 두살이야",
                            "나도 반가워 궁금한거 있으면 물어봐",
                            "잘가 내일도 올거지?",
                            "무슨 말 하는지 모르겠어요",
                            "더울 땐 물을 많이 드세요",]
candidate_input = tokenizer(candidate_text, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt')

candidate_input = {'input_ids': candidate_input['input_ids'].to('cuda'),
                'attention_mask': candidate_input['attention_mask'].to('cuda'),
                'token_type_ids': candidate_input['token_type_ids'].to('cuda'),
                }
                
candidate_embedding = model(**candidate_input, training = False)

while True:
    prompt = input()
    if prompt == 'ㅇㅋ':
        break
    
    context_input = tokenizer(prompt, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt')
    model.to('cpu')
    context_input = model(**context_input, training = False)

    dot_product = torch.matmul(context_input, candidate_embedding.to('cpu').t())
    
    best = torch.argmax(dot_product).item()

    print(candidate_text[best])
    # break

