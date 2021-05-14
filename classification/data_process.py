import os
import json
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import XLNetTokenizer


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_path', default='orig_data')
    arg_parser.add_argument('--temp_data', default='temp_data')
    arg_parser.add_argument('--bert_path', default='/home/lishuan/pretrain_model/chinese_xlnet_base/')
    arg_parser.add_argument('--label_info', default={
        '解除质押': 0, '股份回购': 1, '股东减持': 2, '亏损': 3, '中标': 4, '高管变动': 5, '企业破产': 6,
        '股东增持': 7, '被约谈': 8, '企业收购': 9, '公司上市': 10, '企业融资': 11, '质押': 12
    })
    args = arg_parser.parse_args()
    return args


def data_pro(args):
    train_path = os.path.join(args.data_path, 'duee_fin_train.json')
    dev_path = os.path.join(args.data_path, 'duee_fin_dev.json')

    train_data = data_read(train_path)
    dev_data = data_read(dev_path)
    p(train_data)
    print(len(dev_data))

    train_ids, train_data_info = convert_text_to_id(train_data, args)
    dev_ids, dev_data_info = convert_text_to_id(dev_data, args)

    print(train_data_info, '\n', dev_data_info)

    train_ids_path = os.path.join(args.temp_data, 'xlnet_train_event_ids.json')
    dev_ids_path = os.path.join(args.temp_data, 'xlnet_dev_event_ids.json')

    data_save(train_ids, train_ids_path)
    data_save(dev_ids, dev_ids_path)


def data_read(path):
    text_data = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            text_data.append(json.loads(line))
    return text_data


def convert_text_to_id(text_data, args):
    tokenizer = XLNetTokenizer.from_pretrained(args.bert_path)
    data_ids = []
    data_info = {label_str: 0 for label_str in args.label_info.keys()}
    for sample in tqdm(text_data):
        if 'event_list' not in sample.keys() or 'event_type' not in sample['event_list'][0].keys():
            continue
        input_info = tokenizer.encode_plus(sample['text'], add_special_tokens=True)
        data_ids.append(
            [input_info['input_ids'], input_info['token_type_ids'],
             input_info['attention_mask'], args.label_info[sample['event_list'][0]['event_type']]]
        )
        data_info[sample['event_list'][0]['event_type']] += 1
    return data_ids, data_info


def data_save(data, path):
    with open(path, mode="a", encoding="utf-8") as f:
        json.dump(data, f)
    f.close()


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    args = init_arg_parser()
    data_pro(args)
