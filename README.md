## 本项目主要做了一些常见中文torch版本预训练模型的加载方法

**Bert-wwm, Bert-wmm-ext, Roberta-base, Roberta-large**

```[bash]
from transformers import BertTokenizer, BertModel
bert_path = '预训练模型文件夹的地址'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)

text = '西游记是四大名著之一'

# 第一种convert text to input ids 的方法
# tokens 的内容为['西', '游', '记', '是', '四', '大', '名', '著', '之', '一']
# input_ids 的内容为 [101, 6205, 3952, 6381, 3221, 1724, 1920, 1399, 5865, 722, 671, 102]
# 这种方法给了使用者较大的操作空间，但是token_type_ids和attention_mask需要自己实现
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
attention_mask = [1]*len(input_ids)


# 第二种 convert text to input ids 的方法
# 方便快捷
input_info = tokenizer.encode_plus(text, add_special_tokens=True)
input_ids = input_info['input_ids']
token_type_ids = input_info['token_type_ids']
attention_mask = input_info['attention_mask']

# 版本不一样这里的输出也不一样，这里是新版本transformer的输出
# 老版本的输出如下:
# output_info = bert_model(input_ids=input_ids, attention_mask=attention_mask)
# word_vec = output_info.last_hidden_state
# pool_output = output_info.last_hidden_state
word_vec, pool_ouput = bert_model(input_ids=input_ids, attention_mask=attention_mask)

```


