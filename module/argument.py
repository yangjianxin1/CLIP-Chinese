from dataclasses import dataclass, field

@dataclass
class CLIPArguments:
    """
    自定义的一些参数
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    test_file: str = field(metadata={"help": "测试集"})
    clip_pretrain_path: str = field(metadata={"help": "clip的预训练权重路径"})
    bert_pretrain_path: str = field(default=False, metadata={"help": "bert的预训练权重路径"})
    image_path: str = field(default=False, metadata={"help": "图片存储路径"})
    load_from_bert_clip: bool = field(default=False, metadata={"help": "是否加载BertCLIPModel的预训练权重"})

