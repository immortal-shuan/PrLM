import os
import json
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_path', default='orig_data')
    arg_parser.add_argument('--temp_data', default='temp_data')
    arg_parser.add_argument('--event_data', default='event_data')
    arg_parser.add_argument('--bert_path', default='/home/lishuan/pretrained_model/chinese-roberta-wwm-ext/')
    arg_parser.add_argument('--label_info', default={
        '解除质押': 0, '股份回购': 1, '股东减持': 2, '亏损': 3, '中标': 4, '高管变动': 5, '企业破产': 6,
        '股东增持': 7, '被约谈': 8, '企业收购': 9, '公司上市': 10, '企业融资': 11, '质押': 12
    })
    args = arg_parser.parse_args()
    return args


def data_pro(args):
    train_path = os.path.join(args.data_path, 'duee_fin_train.json')
    dev_path = os.path.join(args.data_path, 'duee_fin_dev.json')
    tag_path = os.path.join(args.data_path, 'duee_fin_event_schema.json')

    train_data = data_read(train_path)
    dev_data = data_read(dev_path)
    tag_data = data_read(tag_path)

    tag_info = tag_pro(tag_data)
    data_save(tag_info, 'temp_data/tag_info.json')
    train_tokens = gen_text_tokens_ids(train_data, tag_info)
    dev_tokens = gen_text_tokens_ids(dev_data, tag_info)

    for label_name in args.label_info.keys():
        train_save_path = os.path.join(args.temp_data, 'train_'+label_name+'_ner_data.json')
        dev_save_path = os.path.join(args.temp_data, 'dev_' + label_name + '_ner_data.json')

        temp_train_data = train_tokens[label_name]
        temp_dev_data = dev_tokens[label_name]

        temp_train_data = delete_rep(temp_train_data)
        temp_dev_data = delete_rep(temp_dev_data)

        data_save(temp_train_data, train_save_path)
        data_save(temp_dev_data, dev_save_path)


def data_read(path):
    text_data = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            text_data.append(json.loads(line))
    return text_data


def tag_pro(tag_data):
    tag_info = {}
    for dict in tag_data:
        tag_info[dict['event_type']] = {}
        tag_info[dict['event_type']]['num_role'] = len(dict['role_list'])
        tag_index = 1
        for sub_dict in dict['role_list']:
            tag_info[dict['event_type']][sub_dict['role']] = [tag_index, tag_index+1]
            tag_index += 2
    return tag_info


def gen_text_tokens_ids(text_data, tag_info):
    data_tokens = {event_type: [] for event_type in tag_info.keys()}
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    for sample in tqdm(text_data):
        if 'event_list' in sample.keys():
            text_tokens = tokenizer.tokenize(sample['text'])
            text_tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
            text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
            tokens_len = len(text_tokens)
            event_list = sample['event_list']

            event2trg_arg = {}
            for event in event_list:
                event_type = event['event_type']
                if event_type in event2trg_arg.keys():
                    trg_tag = event2trg_arg[event_type][0]
                    arg_tag = event2trg_arg[event_type][1]
                else:
                    trg_tag = [0] * tokens_len
                    arg_tag = [0] * tokens_len

                trg_tokens = tokenizer.tokenize(event['trigger'])
                trg_tag = gen_trg_tag(text_tokens, trg_tokens, trg_tag)

                arguments = event['arguments']
                for role_dict in arguments:
                    role = role_dict['role']
                    arg_tokens = tokenizer.tokenize(role_dict['argument'])
                    arg_tag = gen_arg_tag(text_tokens, arg_tokens, arg_tag, tag_info[event_type][role])

                if event_type in event2trg_arg.keys():
                    event2trg_arg[event_type][0] = trg_tag
                    event2trg_arg[event_type][1] = arg_tag
                else:
                    event2trg_arg[event_type] = [trg_tag, arg_tag]
            for event_type_ in event2trg_arg.keys():
                sample_dict = {'text': sample['text'], 'id': sample['id'], 'title': sample['title'], 'text_ids': text_ids}
                sample_dict['trg_tag'] = event2trg_arg[event_type_][0]
                sample_dict['arg_tag'] = event2trg_arg[event_type_][1]
                data_tokens[event_type_].append(sample_dict)
    return data_tokens


def gen_trg_tag(text_tokens, trg_tokens, trg_tag):
    tag_len = len(trg_tokens)
    for i in range(len(text_tokens)-tag_len+1):
        is_i = True
        for j in range(tag_len):
            if trg_tokens[j] != text_tokens[i+j]:
                is_i = False
                break
        if is_i == True:
            trg_tag[i:i+tag_len] = [1]*tag_len
            break
    return trg_tag


def gen_arg_tag(text_tokens, arg_tokens, arg_tag, tag_index):
    tag_len = len(arg_tokens)
    for i in range(len(text_tokens) - tag_len + 1):
        is_i = True
        for j in range(tag_len):
            if arg_tokens[j] != text_tokens[i + j]:
                is_i = False
                break
        if is_i == True:
            if tag_len == 1:
                arg_tag[i] = tag_index[0]
            else:
                arg_tag[i] = tag_index[0]
                arg_tag[i+1:i+tag_len] = [tag_index[1]]*(tag_len-1)
            break
    return arg_tag


def delete_rep(data):
    new_data = []
    for sample in data:
        if sample not in new_data:
            new_data.append(sample)
        else:
            continue
    return new_data


def data_save(data, path):
    with open(path, mode="a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    f.close()


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    args = init_arg_parser()
    data_pro(args)

