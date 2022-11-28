"""
对数据进行过滤
下载的图片中，存在很多默认的图片，对此类图片进行删除，并且把该条数据从训练集中删除，得到新的训练集
"""
import pandas as pd
import os
from os.path import join
import cv2
import numpy as np
from tqdm import tqdm


def filter_default_image(default_file, file):
    default_image = cv2.imread(default_file)
    image = cv2.imread(file)
    delete_flag = False  # 是否要删除的标志
    try:
        difference = cv2.subtract(default_image, image)
        delete_flag = not np.any(difference)  # if difference is all zeros it will return False
    except:
        delete_flag = False

    return delete_flag


def main():
    train_file = './data/train-2.5M.csv'
    image_path = './data/images'
    default_file = './data/train-2.jpg'
    out_file = './data/train-2.5M-filter.csv'
    delete_cnt = 0
    origin_image_cnt = 0

    result = []
    df = pd.read_csv(train_file)
    for _, row in tqdm(df.iterrows()):
        filename = row['filename']
        file = join(image_path, filename)
        if not os.path.exists(file):
            # 不存在该图片，则跳过
            continue
        else:
            origin_image_cnt += 1
            # 图片存在，则查看该图是否为默认图，如果是则删掉该条记录
            delete_flag = filter_default_image(default_file,  file)
            # 如果file是默认图，则删除
            if delete_flag:
                delete_cnt += 1
                print('delete image:{}'.format(file))
                os.remove(file)
            else:
                result.append(row)
    print('origin_image_cnt:{}'.format(origin_image_cnt))
    print('delete_cnt:{}'.format(delete_cnt))
    print('len of filter data:{}'.format(len(result)))
    df = pd.DataFrame(result)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
