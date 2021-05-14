import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange, tqdm
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import AlbertModel
from transformers import XLNetModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('xlnet_large')


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--max_len', default=512)

    arg_parser.add_argument('--stop_num', default=3)
    arg_parser.add_argument('--seed', default=102)
    arg_parser.add_argument('--epoch_num', default=30)
    arg_parser.add_argument('--batch_size', default=8)
    arg_parser.add_argument('--save_model', default=False)
    arg_parser.add_argument('--loss_step', default=8)

    arg_parser.add_argument('--data_path', default='temp_data')
    arg_parser.add_argument('--bert_path', default='/home/lishuan/pretrain_model/chinese_xlnet_large/')
    arg_parser.add_argument('--output_path', default='model_output')

    arg_parser.add_argument('--bert_lr', default=2e-5)
    arg_parser.add_argument('--dropout', default=0.5)
    arg_parser.add_argument('--bert_dim', default=1024)
    arg_parser.add_argument('--num_class', default=13)

    args = arg_parser.parse_args()
    return args


def data_pro(args):
    train_path = os.path.join(args.data_path, 'xlnet_train_event_ids.json')
    dev_path = os.path.join(args.data_path, 'xlnet_dev_event_ids.json')

    train_ids = data_read(train_path)
    dev_ids = data_read(dev_path)
    return train_ids, dev_ids


def data_read(path):
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def Batch(data, args):
    text_input = []
    text_type = []
    text_mask = []
    label = []
    for sample in data:
        text_input.append(sample[0])
        text_type.append(sample[1])
        text_mask.append(sample[2])
        label.append(sample[-1])

    text_input_ = batch_pad(text_input, args, pad=0)
    text_type_ = batch_pad(text_type, args, pad=0)
    text_mask_ = batch_pad(text_mask, args, pad=0)
    return text_input_, text_type_, text_mask_, torch.tensor(label, dtype=torch.long).cuda()


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


class model_classification(nn.Module):
    def __init__(self, args):
        super(model_classification, self).__init__()

        self.bert_model = XLNetModel.from_pretrained(args.bert_path)
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.bert_dim*2, args.num_class)

    def forward(self, input_id, token_type_ids, attention_mask):

        word_vec = self.bert_model(
            input_ids=input_id, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        word_vec = word_vec.last_hidden_state
        max_feature = self.max_pool(word_vec.permute(0, 2, 1)).squeeze(-1)
        avg_feature = self.avg_pool(word_vec.permute(0, 2, 1)).squeeze(-1)
        word_feature = torch.cat((max_feature, avg_feature), dim=-1)
        word_feature = self.dropout(word_feature)
        out = self.fc(word_feature)

        return out


def train(train_data, dev_data, model, optimizer, criterion, args):
    train_len = len(train_data)
    model.zero_grad()

    dev_acc = 0.0
    max_acc_index = 0

    for i in range(args.epoch_num):
        random.shuffle(train_data)

        train_step = 1.0
        train_loss = 0.0

        train_preds = []
        train_labels = []

        for j in trange(0, train_len, args.batch_size):
            model.train()
            if j + args.batch_size < train_len:
                train_batch_data = train_data[j: j+args.batch_size]
            else:
                train_batch_data = train_data[j: train_len]
            text_input, text_type, text_mask, label = Batch(train_batch_data, args)

            out = model(text_input, text_type, text_mask)

            loss = criterion(out, label)
            train_loss += loss.item()

            loss = loss / args.loss_step
            loss.backward()

            if int(train_step % args.loss_step) == 0:
                optimizer.step()
                model.zero_grad()

            pred = out.argmax(dim=-1).cpu().tolist()
            train_preds.extend(pred)
            train_labels.extend(label.cpu().tolist())
            train_step += 1.0

        train_acc = accuracy_score(np.array(train_preds), np.array(train_labels))

        print('epoch:{}\n train_loss:{}\n train_acc:{}'.format(i, train_loss / train_step, train_acc))

        dev_acc_ = dev(model=model, dev_data=dev_data, args=args)

        if dev_acc <= dev_acc_:
            dev_acc = dev_acc_
            max_acc_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}.pth'.format( i))
                torch.save(model.state_dict(), save_file)

        if i - max_acc_index > args.stop_num:
            break

    file = open('lijie00_result.txt', 'a')
    file.write('max_acc: {}, {}'.format(max_acc_index, dev_acc) + '\n')
    file.close()

    print('-----------------------------------------------------------------------------------------------------------')
    print('max_acc: {}, {}'.format(max_acc_index, dev_acc))
    print('-----------------------------------------------------------------------------------------------------------')


def dev(model, dev_data, args):
    model.eval()

    dev_len = len(dev_data)

    dev_preds = []
    dev_labels = []

    with torch.no_grad():
        for m in trange(0, dev_len, args.batch_size):
            if m + args.batch_size < dev_len:
                dev_batch_data = dev_data[m: m+args.batch_size]
            else:
                dev_batch_data = dev_data[m: dev_len]
            text_input, text_type, text_mask, label = Batch(dev_batch_data, args)

            out = model(text_input, text_type, text_mask)

            pred = out.argmax(dim=-1).cpu().tolist()
            dev_preds.extend(pred)
            dev_labels.extend(label.cpu().tolist())

    dev_acc = accuracy_score(np.array(dev_preds), np.array(dev_labels))
    dev_pre = precision_score(np.array(dev_labels), np.array(dev_preds), average='macro')
    dev_rec = recall_score(np.array(dev_labels), np.array(dev_preds), average='macro')
    dev_f1 = f1_score(np.array(dev_labels), np.array(dev_preds), average='macro')

    dev_pre_avg = precision_score(np.array(dev_labels), np.array(dev_preds), average=None)
    dev_rec_avg = recall_score(np.array(dev_labels), np.array(dev_preds), average=None)
    dev_f1_avg = f1_score(np.array(dev_labels), np.array(dev_preds), average=None)

    print('dev_acc:{} \ndev_pre:{} \ndev_rec:{} \n dev_f1:{}'.format(dev_acc, dev_pre, dev_rec, dev_f1))
    print('dev_pre_avg:', dev_pre_avg)
    print('dev_rec_avg:', dev_rec_avg)
    print('dev_f1_avg:', dev_f1_avg)
    return dev_acc


def main():
    args = init_arg_parser()
    setup_seed(args)

    train_ids, dev_ids = data_pro(args)

    model = model_classification(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.bert_lr)
    criterion = torch.nn.CrossEntropyLoss()

    train(train_ids, dev_ids, model, optimizer, criterion, args)


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    args = init_arg_parser()
    main()


