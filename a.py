with open('./outputs/poly220929_0811_bs128_ep5_data21000000_en.txt', 'r') as f:
    with open('./outputs/poly220929_0811_bs128_ep5_data21000000_en_.txt', 'w') as f_:
        for line in f.readlines():
            if line != 'torch.Size([128, 128])\n':
                f_.write(line)