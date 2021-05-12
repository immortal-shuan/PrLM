## 本项目主要做了一些常见中文torch版本预训练模型的加载方法

**Bert-wwm, Bert-wmm-ext, Roberta-base, Roberta-large，ernie-1.0**

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
# pool_output = output_info.pooler_output
word_vec, pool_ouput = bert_model(input_ids=input_ids, attention_mask=attention_mask)

```

其中需要注意的是ernie的词表（即vocab.txt文件）和Bert-wwm, Bert-wmm-ext, Roberta-base, Roberta-large的词表文件不一样，即通过Berttokenizer加载Bert-wwm词表生成的input_ids可以在Bert-wmm-ext, Roberta-base, Roberta-large内通用，不能够在ernie里面使用，ernie的input_ids需要单独加载ernie词表生成input_ids.

**Electra**

Electra的vocab与Bert-wwm一致，tokenizer可以直接用上面的convert text to ids 的方法
也可以把BertTokenizer替换成ElectraTokenizer进行转换

模型的加载方法如下：

```[bash]
from transformers import ElectraModel
bert_path = '预训练模型文件夹的地址'
bert_model = ElectraModel.from_pretrained(bert_path)

# 版本不一样这里的输出也不一样，这里是新版本transformer的输出
# 老版本的输出如下:
# output_info = bert_model(input_ids=input_ids, attention_mask=attention_mask)
# word_vec = output_info.last_hidden_state
word_vec = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

```

**Albert**

Albert的vocab与Bert-wwm一致，tokenizer可以直接用上面的convert text to ids 的方法
也可以把BertTokenizer替换成ElectraTokenizer进行转换

模型的加载方法如下：

```[bash]
from transformers import AlbertModel
bert_path = '预训练模型文件夹的地址'
bert_model = AlbertModel.from_pretrained(bert_path)

# 版本不一样这里的输出也不一样，这里是新版本transformer的输出
# 老版本的输出如下:
# output_info = bert_model(input_ids=input_ids, attention_mask=attention_mask)
# word_vec = output_info.last_hidden_state
# pool_output = output_info.pooler_output
word_vec, pool_ouput = bert_model(input_ids=input_ids, attention_mask=attention_mask)

```

**XLNet**

```[bash]
from transformers import XLNetTokenizer, XLNetModel
bert_path = '预训练模型文件夹的地址'
tokenizer = XLNetTokenizer.from_pretrained(bert_path)
bert_model = XLNetModel.from_pretrained(bert_path)

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
# pool_output = output_info.pooler_output
word_vec, pool_ouput = bert_model(input_ids=input_ids, attention_mask=attention_mask)

```

注：Xlnet在处理text文档的时候，需要额外在环境中配置sentencepiece包

```[bash]
pip install sentencepiece
```

**Macbert**

