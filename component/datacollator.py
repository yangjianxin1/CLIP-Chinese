from typing import Any, Dict, List
import torch


class CLIPCollator(object):
    def __init__(self, clip_processor, max_seq_length):
        self.clip_processor = clip_processor
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts, pixel_values_list = [], []
        for data in features:
            # 如果图片预处理失败，则跳过该图片
            if data['pixel_values'] is None:
                continue
            texts.append(data['text'])
            pixel_values_list.append(data['pixel_values'])
        # 进行tokenize
        inputs = self.clip_processor(
            text=texts, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        )
        pixel_values_list = torch.concat(pixel_values_list, dim=0)
        inputs['return_loss'] = True
        inputs['pixel_values'] = pixel_values_list
        inputs.pop('token_type_ids')
        return inputs
