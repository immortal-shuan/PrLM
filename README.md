## 本项目主要做了一些常见中文torch版本预训练模型的加载方法

**Bert-wwm, Bert-wmm-ext, Roberta-base, Roberta-large**

```[bash]
from transformers import BertTokenizer, BertModel
bert_path = '预训练模型文件夹的地址'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)

text = '西游记是四大名著之一'

# 第一种convert text to input ids 的方法
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

```
