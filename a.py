with open('./outputs/bi220922_1800_bs256_ep5_data1363581.txt', 'r') as f:
    with open('./outputs/bi220922_1800_bs256_ep5_data1363581_.txt', 'w') as f_:
        for i in f.readlines():
            if i != "\n":
                f_.write(i)