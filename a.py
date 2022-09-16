import json

context, candidate = [], []
filename = "./data/original_data/personachat/personachat_truecased_full_train.json"
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

cnt = 0
for i in range(len(data)):
    dataset = data[i]['utterances'][-1]['history']

    if len(dataset) > 1:
        if len(dataset) % 2 == 0:
            context += dataset[0::2]
            candidate += dataset[1::2]
        elif len(dataset) % 2 == 1:
            cnt += 1
            context += dataset[0:len(dataset)-1:2]
            candidate += dataset[1::2]

for i in range(7):
    print(context[i])
    print('     ', candidate[i])

print(cnt)
print(len(context))
print(cnt + len(context))