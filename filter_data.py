"""
对训练数据进行过滤
下载的图片中，存在gif图片，对此类图片进行删除，并且把该条数据从训练集中删除，得到过滤后的训练集
"""
import pandas as pd
import os
from os.path import join
from tqdm import tqdm
import imghdr


def main():
    train_file = './data/train.csv'
    image_path = './data/images'
    out_file = './data/train-filter.csv'

    result = []
    df = pd.read_csv(train_file)
    for _, row in tqdm(df.iterrows()):
        filename = row['filename']
        file = join(image_path, filename)

        # 如果存在该图片
        if os.path.exists(file):
            # 判断图片是否为gif图或者损坏
            img_type = imghdr.what(file)
            # 图片损坏，或者为gif图，则跳过
            if img_type is None or img_type == 'gif':
                print('remove file:{}'.format(file))
                os.remove(file)
            else:
                result.append(row)
    print('len of filter data:{}'.format(len(result)))
    df = pd.DataFrame(result)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
