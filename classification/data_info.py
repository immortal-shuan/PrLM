import os
import json
import argparse
from tqdm import tqdm


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_path', default='orig_data')
    arg_parser.add_argument('--temp_data', default='temp_data')
    arg_parser.add_argument('--bert_path', default='F:/pre_trained model/roberta_wwm')
    arg_parser.add_argument('--label_info', default={
        '解除质押': 0, '股份回购': 1, '股东减持': 2, '亏损': 3, '中标': 4, '高管变动': 5, '企业破产': 6,
        '股东增持': 7, '被约谈': 8, '企业收购': 9, '公司上市': 10, '企业融资': 11, '质押': 12
    })
    args = arg_parser.parse_args()
    return args


def data_read(path):
    text_data = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            text_data.append(json.loads(line))
    return text_data


def text_info_get(data, args):
    text_len_info = {_: [] for _ in args.label_info.keys()}
    min_len = 10000
    max_len = 0
    for sample in tqdm(data):
        text_len = len(sample['text'])
        if text_len > max_len:
            max_len = text_len
        if text_len < min_len:
            min_len = text_len
        if 'event_list' in sample.keys():
            event_list = sample['event_list']
            for event in event_list:
                event_type = event['event_type']
                text_len_info[event_type].append(text_len)
        else:
            continue
    return min_len, max_len, text_len_info


def gen_lens_num(min_num, num_thre, thre, texts_len):
    text_num = len(texts_len)
    thre2num = {i: 0 for i in range(num_thre)}
    for len_ in tqdm(texts_len):
        for i in range(9):
            if min_num + i * thre <= len_ < min_num + (i + 1) * thre:
                thre2num[i] += 1
                break
    threlist = ['{}_{}'.format(min_num + i * thre, min_num + (i + 1) * thre) for i in thre2num.keys()]
    numlist = [num for num in thre2num.values()]
    return text_num, threlist, numlist



args = init_arg_parser()
train_path = os.path.join(args.data_path, 'duee_fin_train.json')
dev_path = os.path.join(args.data_path, 'duee_fin_dev.json')
train_data = data_read(train_path)
dev_data = data_read(dev_path)

train_min_len, train_max_len, train_text_len_info = text_info_get(train_data, args)
dev_min_len, dev_max_len, dev_text_len_info = text_info_get(dev_data, args)

for name in args.label_info.keys():
    print(name)
    a, b, c = gen_lens_num(0, 7, 512, train_text_len_info[name])
    print(a, c)
    print('------------------------------------------------------------------------------------------------------------')









