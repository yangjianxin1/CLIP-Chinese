"""
从BertCLIP模型中，拎出bert的权重，单独保存
"""
from component.model import BertCLIPTextModel
from transformers import BertTokenizerFast

if __name__ == '__main__':
    model_name_or_path = 'output/clip/checkpoint-final'
    save_path = 'output/bert'
    text_model = BertCLIPTextModel.from_pretrained(model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
    text_model.text_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
