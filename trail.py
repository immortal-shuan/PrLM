import json


def data_read(path):
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


path = 'data_ids/ThucNews_ids.json'
train_ids = data_read(path)


def p(data, num=5):
    for i in range(num):
        print(data[i])


p(train_ids)
