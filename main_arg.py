import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from tqdm import trange, tqdm
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import AlbertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--max_len', default=512)
    arg_parser.add_argument('--window_size', default=256)

    arg_parser.add_argument('--stop_num', default=3)
    arg_parser.add_argument('--seed', default=102)
    arg_parser.add_argument('--epoch_num', default=30)
    arg_parser.add_argument('--save_model', default=False)
    arg_parser.add_argument('--loss_step', default=1)
    arg_parser.add_argument('--label_name', default='质押')

    arg_parser.add_argument('--data_path', default='temp_data')
    arg_parser.add_argument('--bert_path', default='/home/lishuan/pretrain_model/chinese-electra-large/')
    arg_parser.add_argument('--output_path', default='model_output')
    arg_parser.add_argument('--result', default='result')

    arg_parser.add_argument('--bert_lr', default=2e-5)
    arg_parser.add_argument('--crf_lr', default=1e-3)
    arg_parser.add_argument('--dropout', default=0.5)
    arg_parser.add_argument('--bert_dim', default=1024)
    arg_parser.add_argument('--num_class', default=19)

    args = arg_parser.parse_args()
    return args


def data_pro(args):
    train_path = os.path.join(args.data_path, 'train_'+args.label_name+'_ner_data.json')
    dev_path = os.path.join(args.data_path, 'dev_' + args.label_name + '_ner_data.json')
    print(train_path)
    train_ids = data_read(train_path)
    dev_ids = data_read(dev_path)
    return train_ids, dev_ids


def data_read(path):
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def sample_pro(sample, args):
    text_len = len(sample[0])
    if text_len <= args.max_len:
        input_ids = torch.tensor([sample[1]], dtype=torch.long).cuda()
        attention_mask = torch.tensor([[1]*text_len], dtype=torch.long).cuda()
        trg_tag = torch.tensor(sample[2], dtype=torch.long).cuda()
        arg_tag = torch.tensor(sample[3], dtype=torch.long).cuda()
        return input_ids, attention_mask, trg_tag, arg_tag
    else:
        input_ids = []
        attention_mask = []
        for i in range(0, text_len, args.max_len):
            input_ids.append(sample[1][i:i+args.max_len])
            attention_mask.append([1]*len(sample[1][i:i+args.max_len]))
            if i+args.max_len >= text_len:
                break
        input_ids = batch_pad(input_ids, args)
        attention_mask = batch_pad(attention_mask, args)

        num_sub_text = text_len // args.max_len
        is_iter = text_len % args.max_len

        if is_iter == 0:
            trg_tag = torch.tensor(sample[2] + [0] * (num_sub_text * args.max_len - text_len),
                                   dtype=torch.long).cuda()
            arg_tag = torch.tensor(sample[3] + [0] * (num_sub_text * args.max_len - text_len),
                                   dtype=torch.long).cuda()
        else:
            trg_tag = torch.tensor(sample[2] + [0] * ((num_sub_text + 1) * args.max_len - text_len),
                                   dtype=torch.long).cuda()
            arg_tag = torch.tensor(sample[3] + [0] * ((num_sub_text + 1) * args.max_len - text_len),
                                   dtype=torch.long).cuda()
        return input_ids, attention_mask, trg_tag, arg_tag


def batch_pad(batch_data, args, pad=0):
    seq_len = [len(i) for i in batch_data]
    max_len = max(seq_len)
    if max_len > args.max_len:
        max_len = args.max_len
    out = []
    for line in batch_data:
        if len(line) < max_len:
            out.append(line + [pad] * (max_len - len(line)))
        else:
            out.append(line[:args.max_len])
    return torch.tensor(out, dtype=torch.long).cuda()


# def sub_sample_combine(batch_data):
#     b, s, d = batch_data.shape
#     new_batch_data = batch_data[0]
#     for i in range(1, b):
#         new_batch_data[-256:] += batch_data[i][:256]
#         new_batch_data[-256:] = new_batch_data[-256:] / 2.0
#         new_batch_data = torch.cat((new_batch_data, batch_data[i][256:512]))
#     return new_batch_data


class model_ner(nn.Module):
    def __init__(self, args):
        super(model_ner, self).__init__()

        self.bert_model = ElectraModel.from_pretrained(args.bert_path)
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.bert_dim, args.num_class)
        self.crf = CRF(args.num_class, batch_first=True)

    def forward(self, input_id, attention_mask, tag):
        word_vec = self.bert_model(
            input_ids=input_id, attention_mask=attention_mask
        )
        word_vec = word_vec.last_hidden_state
        b, s, d = word_vec.shape
        if b != 1:
            word_vec = word_vec.reshape(1, -1, d)
        pred_prob = self.linear(self.dropout(word_vec))
        tag = tag.reshape(1, -1)
        loss = self.crf(pred_prob, tag)
        return loss * -1.0

    def decode(self, input_id, attention_mask):
        word_vec = self.bert_model(
            input_ids=input_id, attention_mask=attention_mask
        )
        word_vec = word_vec.last_hidden_state
        b, s, d = word_vec.shape
        if b != 1:
            word_vec = word_vec.reshape(1, -1, d)
        pred_prob = self.linear(self.dropout(word_vec))
        pred_tag = self.crf.decode(pred_prob)
        return pred_tag


def train(train_data, dev_data, model, optimizer, args):
    train_len = len(train_data)
    model.zero_grad()

    max_index = 0

    final_dev_loss = 1000.0

    for i in range(args.epoch_num):
        random.shuffle(train_data)

        train_step = 1.0
        train_loss = 0.0

        for j in trange(0, train_len):
            model.train()
            sample = train_data[j]

            input_ids, attention_mask, trg_tag, arg_tag = sample_pro(sample, args)

            loss = model(input_ids, attention_mask, arg_tag)

            train_loss += loss.item()

            loss = loss / args.loss_step
            loss.backward()

            if int(train_step % args.loss_step) == 0:
                optimizer.step()
                model.zero_grad()

        dev_loss = dev(model=model, dev_data=dev_data, index=i,  args=args)
        print('epoch:{}'.format(i), '\n', train_loss/train_len, dev_loss)

        if dev_loss < final_dev_loss:
            final_dev_loss = dev_loss
            max_index = i

        if i - max_index > args.stop_num:
            break
    print('--------------------------------------------------------------')
    print(max_index, '\n', final_dev_loss)
    print('--------------------------------------------------------------')
    return max_index, final_dev_loss


def dev(model, dev_data, index, args):
    model.eval()

    dev_len = len(dev_data)
    dev_loss = 0.0

    with torch.no_grad():
        temp_path = os.path.join(args.result, 'electra-large_' + args.label_name+'_{}'.format(index)+'.json')
        with open(temp_path, mode="a", encoding='utf-8') as f:
            for i in range(dev_len):
                sample = dev_data[i]
                input_ids, attention_mask, trg_tag, arg_tag = sample_pro(sample, args)
                loss = model(input_ids, attention_mask, arg_tag)
                pred_tag = model.decode(input_ids, attention_mask)

                f.write(json.dumps([sample[3], pred_tag[0]], ensure_ascii=False))
                f.write('\n')

                dev_loss += loss.item()
            f.close()
    return dev_loss / dev_len


def main(args):
    setup_seed(args)

    train_data, dev_data = data_pro(args)

    model = model_ner(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    fc_para = list(map(id, model.linear.parameters()))
    crf_para = list(map(id, model.crf.parameters()))
    base_para = filter(lambda p: id(p) not in fc_para+crf_para, model.parameters())
    params = [
        {'params': base_para},
        {'params': model.linear.parameters(), 'lr': args.crf_lr},
        {'params': model.crf.parameters(), 'lr': args.crf_lr}
    ]
    optimizer = Adam(params, lr=args.bert_lr)

    max_index, final_dev_loss = train(train_data, dev_data, model, optimizer, args)
    return max_index, final_dev_loss


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    args = init_arg_parser()
    label_names = ['解除质押', '股份回购', '股东减持', '亏损', '中标', '高管变动', '企业破产',
                  '股东增持', '被约谈', '企业收购', '公司上市', '企业融资', '质押']
    num_classes = [19, 15, 19, 11, 13, 17, 11,
                   19, 9, 13, 17, 15, 19]
    max_indexes = []
    for i in range(len(label_names)):
        args.label_name = label_names[i]
        args.num_class = num_classes[i]
        print(args.label_name)
        max_index, final_dev_loss = main(args)
        max_indexes.append(max_index)
        print(max_index, final_dev_loss)
    print(max_indexes)
    print(args.bert_path)




