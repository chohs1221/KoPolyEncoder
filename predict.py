from kobert_tokenizer import KoBERTTokenizer
import torch

from tqdm import tqdm

from modeling import BiEncoder
from parsing import pickling
from utils import empty_cuda_cache

valid_context, valid_candidate = pickling('./data/pickle/valid.pickle', 'load')
valid_context += valid_candidate

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BiEncoder.from_pretrained('./checkpoints/firstmodel_ep1')
model.to('cuda');
model.eval()

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
candidate_embeddings = model(**candidate_input, training = False)

# print(len(valid_context))
# try:
#     candidate_embeddings = pickling('./data/pickle/candidates512.pickle', act='load')
# except:
#     candidate_inputs = []
#     batch_size = 512
#     for i in tqdm(range(0, len(valid_context)-batch_size, batch_size)):
#         candidate_input = tokenizer(valid_context[i: i+256], padding='max_length', max_length=50, truncation=True, return_tensors = 'pt')

#         candidate_input = {'input_ids': candidate_input['input_ids'].to('cuda'),
#                         'attention_mask': candidate_input['attention_mask'].to('cuda'),
#                         'token_type_ids': candidate_input['token_type_ids'].to('cuda'),
#                         }
#         candidate_inputs.append(candidate_input)
        
#     print(len(candidate_inputs))
#     candidate_embeddings = torch.Tensor().to('cuda')
#     for candidate_input in tqdm(candidate_inputs):
#         empty_cuda_cache()
#         with torch.no_grad():
#             candidate_embedding = model(**candidate_input, training = False)
#             candidate_embeddings = torch.cat([candidate_embeddings, candidate_embedding], dim=0)
#             # print(candidate_embeddings.shape)

#     pickling('./data/pickle/candidates512.pickle', act='save', data=candidate_embeddings)

print(candidate_embeddings.shape)

while True:
    prompt = input()
    if prompt == 'okay':
        break
    
    context_input = tokenizer(prompt, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt')
    context_input = {'input_ids': context_input['input_ids'].to('cuda'),
                    'attention_mask': context_input['attention_mask'].to('cuda'),
                    'token_type_ids': context_input['token_type_ids'].to('cuda'),
                    }

    model.to('cuda')
    context_output = model(**context_input, training = False)

    dot_product = torch.matmul(context_output, candidate_embeddings.to('cuda').t())
    
    best = torch.argmax(dot_product).item()

    print(candidate_text[best])
    # break

