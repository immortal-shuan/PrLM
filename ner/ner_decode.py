import json
from tqdm import tqdm

def decode_pro(path):
    label_data, pred_data = data_read(path)
    pre_score, rec_score, f1_score = dev_score(label_data, pred_data)
    return pre_score, rec_score, f1_score



def data_read(path):
    label_data = []
    pred_data = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f):
                label_data.append(json.loads(line)[0])
                pred_data.append(json.loads(line)[1])
    return label_data, pred_data


def dev_score(labels, preds):
    pre_res = []
    rec_res = []

    assert len(labels) == len(preds)

    for i in range(len(labels)):
        sub_pre_res = PreScore(labels[i], preds[i])
        sub_rec_res = RecallScore(labels[i], preds[i])

        pre_res.extend(sub_pre_res)
        rec_res.extend(sub_rec_res)

    pre_score = float(sum(pre_res)) / float(len(pre_res))
    rec_score = float(sum(rec_res)) / float(len(rec_res))
    f1_score = 2.0 * pre_score * rec_score / (pre_score + rec_score)

    return pre_score, rec_score, f1_score


def PreScore(labels, preds):
    start_index = [i for i in range(1, 18, 2)]
    text_len = len(labels)
    res = []
    for i in range(text_len):
        preds_tag = preds[i]
        if preds_tag in start_index:
            real_tag = labels[i]
            if real_tag == preds_tag:
                Is_true = 1
                if preds[i+1] == real_tag + 1 and labels[i+1] == real_tag + 1:
                    i += 1
                    while i < text_len:
                        if preds[i+1] == 0 or labels[i+1] == 0:
                            break
                        elif preds[i+1] == real_tag + 1 and labels[i+1] == real_tag + 1:
                            i += 1
                        else:
                            Is_true = 0
                            break
                    res.append(Is_true)
                elif preds[i+1] == 0 and labels[i+1] == 0:
                    res.append(1)
                else:
                    res.append(0)

            else:
                res.append(0)
    return res


def RecallScore(labels, preds):
    start_index = [i for i in range(1, 18, 2)]
    text_len = len(labels)
    res = []
    for i in range(text_len):
        real_tag = labels[i]
        if real_tag in start_index:
            preds_tag = preds[i]
            if real_tag == preds_tag:
                Is_true = 1
                if preds[i + 1] == real_tag + 1 and labels[i + 1] == real_tag + 1:
                    i += 1
                    while i < text_len:
                        if preds[i + 1] == 0 or labels[i + 1] == 0:
                            break
                        elif preds[i + 1] == real_tag + 1 and labels[i + 1] == real_tag + 1:
                            i += 1
                        else:
                            Is_true = 0
                            break
                    res.append(Is_true)
                elif preds[i + 1] == 0 and labels[i + 1] == 0:
                    res.append(1)
                else:
                    res.append(0)

            else:
                res.append(0)
    return res


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    model_name = 'electra-180g-large'
    label_name = ['解除质押', '股份回购', '股东减持', '亏损', '中标', '高管变动', '企业破产',
                  '股东增持', '被约谈', '企业收购', '公司上市', '企业融资', '质押']
    max_score_index = [2, 0, 1, 1, 1, 1, 2, 2, 4, 1, 3, 1, 3]
    res = []
    total_label_data = []
    total_pred_data = []
    for i in range(13):
        path = 'result/{}_{}_{}.json'.format(model_name, label_name[i], max_score_index[i])
        label_data, pred_data = data_read(path)
        total_label_data.extend(label_data)
        total_pred_data.extend(pred_data)
        pre_score, rec_score, f1_score = dev_score(label_data, pred_data)
        res.append([pre_score, rec_score, f1_score])

    print(res)
    pre_score_, rec_score_, f1_score_ = dev_score(total_label_data, total_pred_data)
    print(pre_score_, rec_score_, f1_score_)