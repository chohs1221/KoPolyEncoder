import csv

with open('./data/original_data/ubuntu2/test.csv', 'r', encoding='utf-8') as f:
    data = csv.reader(f)

    dataset = []
    for i, d in enumerate(data):
        if i == 0:
            continue
        print(d)
        print(len(d))
        print(len(d[0]))
        break