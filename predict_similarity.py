"""
计算图文相似度，以及文本相似度的脚本
"""
import torch

from component.model import BertCLIPModel
from transformers import CLIPProcessor
from PIL import Image


def load_model_and_processor(model_name_or_path):
    # 加载模型
    model = BertCLIPModel.from_pretrained(model_name_or_path)
    # note: 代码库默认使用CLIPTokenizer, 这里需要设置自己需要的tokenizer的名称
    CLIPProcessor.tokenizer_class = 'BertTokenizerFast'
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    return model, processor


def process_data(texts, image_files, clip_processor):
    # 如果存在需要对图片进行预处理，则读取文件
    if image_files is not None:
        images = [Image.open(x).convert('RGB') for x in image_files]
    else:
        images = None
    # 预处理
    inputs = clip_processor(images=images, text=texts, return_tensors='pt', padding=True)
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    return inputs


def cal_image_text_sim(model, clip_processor):
    """
    计算图片和所有候选文本之间的相似度
    """
    print('-------------- 计算图文相似度 --------------')
    texts = [
        '秋天跑车唯美图片桌面壁纸', '可爱的小鸡', '一群可爱的小黄鸡在篮子里', '一只小狗', '一只可爱的小猫', '清澈的湖水，蓝蓝的天空，茂密的树木',
        '冬日里，一只老虎在雪地玩耍', '一只老虎在河边喝水', '一辆公交车停在路边', '一只公鸡在打鸣'
    ]
    image_files = [
        './images/test/autumn_car.jpeg', './images/test/bus.jpeg', './images/test/cat.jpeg', './images/test/cock.jpeg',
        './images/test/cute_chick.jpeg', './images/test/dog.jpeg', './images/test/lake_tree.jpeg',
        './images/test/tiger.jpeg', './images/test/tiger_river.jpeg'
    ]
    # 特征处理
    inputs = process_data(texts, image_files, clip_processor)

    with torch.no_grad():
        out = model(**inputs)

    # 对于每张图片，其与所有文本的相似度
    logits_per_image = out.logits_per_image
    # 对分数做softmax
    logits_per_image = torch.softmax(logits_per_image, dim=-1)
    # 对于每张图片，将其与所有文本的相似度，进行降序排序
    logits_per_image = logits_per_image.numpy().tolist()
    for scores, file in zip(logits_per_image, image_files):
        result = sorted([(text, score) for text, score in zip(texts, scores)], key=lambda x: x[1], reverse=True)
        print('图片和所有候选文本的相似度：{}'.format(file))
        print(result)
        print()


def cal_text_text_sim(model, clip_processor):
    """
    计算文本和文本之间的相似度
    """
    print('-------------- 计算文本相似度 --------------')
    texts = [
        '桑巴军团', '巴西', '日耳曼战车',
        '德国', '一个圆圆的月亮高高挂在天空', '夜幕中的白玉盘升起，星光灿烂', '小猪', '佩奇', '足球场', '绿茵',
        '雪花漫天飞舞，狂风怒号，天地之间白茫茫一片', '北国风光，千里冰封，万里雪飘，银装素裹',
        '大漠沙如雪，燕山月似钩', '月光洒在沙滩上，就像铺上了一层白皑皑的雪。燕山上，月亮像钩子一般',
        '天街小雨润如酥，草色摇看近却无', '长安街上细密的春雨润滑如酥，远望草色连成一片，近看却又显得稀疏',
        '一只老虎在草原上追捕一只小鹿', '大猫在飞速狂奔，捕杀猎物', '英雄联盟', 'lol'
    ]
    inputs = process_data(texts, None, clip_processor)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
        # normalized features
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # 计算两两文本之间的相似度
        logit_scale = model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, text_embeds.t()) * logit_scale

    # 对于每个文本，将自己的分数置为-10000
    batch_size = logits_per_text.size(0)
    eye = torch.eye(batch_size) * -10000
    logits_per_text = logits_per_text + eye
    # 对分数做softmax
    logits_per_text = torch.softmax(logits_per_text, dim=-1)

    # 对于每个文本，将其与所有文本的相似度，进行降序排序
    logits_per_text = logits_per_text.numpy().tolist()
    for scores, text in zip(logits_per_text, texts):
        result = sorted([(text, score) for text, score in zip(texts, scores)], key=lambda x: x[1], reverse=True)
        print('文本和所有候选文本的相似度：{}'.format(text))
        print(result)
        print()


def cal_image_image_sim(model, clip_processor):
    """
    计算图片与图片之间的相似度
    """
    print('-------------- 计算图图相似度 --------------')
    image_files = [
        './images/test/bus.jpeg', './images/test/bus2.jpeg',
        './images/test/cat.jpeg', './images/test/cat2.jpeg', './images/test/cock.jpeg',
        './images/test/cute_chick.jpeg', './images/test/dog.jpeg', './images/test/dog2.jpeg',
        './images/test/tiger.jpeg', './images/test/tiger_river.jpeg', './images/test/autumn_car.jpeg'
    ]
    # 特征处理
    inputs = process_data(None, image_files, clip_processor)

    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        # 计算两两文本之间的相似度
        logit_scale = model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, image_embeds.t()) * logit_scale

    # 对于每个文本，将自己的分数置为-10000
    batch_size = logits_per_image.size(0)
    eye = torch.eye(batch_size) * -10000
    logits_per_image = logits_per_image + eye
    # 对分数做softmax
    logits_per_image = torch.softmax(logits_per_image, dim=-1)

    # 对于每个文本，将其与所有文本的相似度，进行降序排序
    logits_per_image = logits_per_image.numpy().tolist()
    for scores, image in zip(logits_per_image, image_files):
        result = sorted([(image, score) for image, score in zip(image_files, scores)], key=lambda x: x[1], reverse=True)
        print('图片和所有候选图片的相似度：{}'.format(image))
        print(result)
        print()


def main():
    model_name_or_path = 'YeungNLP/clip-vit-bert-chinese-1M'
    # 加载模型
    model, clip_processor = load_model_and_processor(model_name_or_path)
    # 预测相似度
    cal_image_text_sim(model, clip_processor)
    cal_text_text_sim(model, clip_processor)
    cal_image_image_sim(model, clip_processor)


if __name__ == '__main__':
    main()

