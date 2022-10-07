def make_testset(filename):
    your_context, your_candidate = [], []
    partners_context, partners_candidate = [], []
    with open(filename, 'r') as f:
        datasets = f.readlines()
        
        for dataset in datasets:
            if dataset.strip().split()[2] == 'persona:':
                if dataset.strip().split()[0] == '1':
                    your_history = ' '.join(dataset.strip().split()[1:]) + '\n'
                    partners_history = 'your '

                    flag = False
                    if partners_context:
                        partners_context.pop(-1)

                else:
                    if dataset.strip().split()[1] == 'your':
                        your_history += ' '.join(dataset.strip().split()[1:]) + '\n'
                    elif dataset.strip().split()[1] == "partner's":
                        partners_history += ' '.join(dataset.strip().split()[2:]) + '\n'
            else:
                your_history += dataset[dataset.index(' ') + 1:].strip().split('\t')[0] + ' '
                your_context.append(your_history)
                your_history += dataset[dataset.index(' ') + 1:].strip().split('\t')[1] + ' '
                your_candidate.append(dataset[dataset.index(' ') + 1:].strip().split('\t')[1])

                partners_history += dataset[dataset.index(' ') + 1:].strip().split('\t')[0] + ' ' + dataset[dataset.index(' ') + 1:].strip().split('\t')[1] + ' '
                partners_context.append(partners_history)
                if flag:
                    partners_candidate.append(dataset[dataset.index(' ') + 1:].strip().split('\t')[0])
                else:
                    flag = True

        partners_context.pop(-1)
        
        context = your_context + partners_context
        candidate = your_candidate + partners_candidate

        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate

a, b = make_testset("./data/original_data/personachat/train_both_original.txt")
print('--------------------------------------')
idx = 1
for idx in range(0, 5):
    print(a[idx])
    print('--------------------------------------')
    print(b[idx])
    print('--------------------------------------')
print(len(a), len(b))