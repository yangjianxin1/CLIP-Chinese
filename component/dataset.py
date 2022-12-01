from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import requests
from loguru import logger
from glob import glob
from os.path import join
from tqdm import tqdm


class CLIPDataset(Dataset):

    def __init__(self, file, clip_processor, image_path):
        df = pd.read_csv(file, usecols=['text', 'filename'])
        data_list = df.to_dict('records')
        print('len of data:{}'.format(len(data_list)))
        self.data_list = data_list
        self.clip_processor = clip_processor
        self.image_path = image_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        row = self.data_list[index]
        text = row['text'].strip()
        filename = row['filename']
        file = join(self.image_path, filename)
        try:
            image = Image.open(file).convert('RGB')
        except Exception as e:
            # 下载图片失败
            logger.info('open image error, text: {}, filename:{}'.format(text, filename))
            logger.info(e)
            image = None

        if image is None:
            pixel_values = None
        else:
            pixel_values = self.clip_processor(images=image, return_tensors='pt')['pixel_values']
        data = {'pixel_values': pixel_values, 'text': text}
        return data
