cnt = 0
with open('./data/original_data/personachat/test_none_original_.txt', 'r') as f:
    datas = f.readlines()
    for data in datas[:1]:
        print(len(data.strip().split('\t')[3].split('|')))
