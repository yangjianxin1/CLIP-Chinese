import requests
import pandas as pd
import os
from os.path import join
from tqdm import tqdm
import multiprocessing
from multiprocessing import cpu_count
from loguru import logger

"""
下载图片
"""


def download(file, url):
    try:
        headers = {"User-Agent": "Chrome/68.0.3440.106"}
        response = requests.get(url, headers=headers)
        status_code = response.status_code
        content_type = response.headers.get('Content-Type')
        # 请求成功并且不是gif图
        if status_code == 200 and content_type != 'image/gif':
            image = response.content
            with open(file, 'wb') as f:
                f.write(image)
    except Exception as e:
        # 下载图片失败
        logger.info('downloading image error, url:{}'.format(url))
        logger.info(e)


def main():
    thread_num = 20  # 线程数量
    in_file = './data/train.csv'
    out_path = './data/images'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    df = pd.read_csv(in_file)
    print(len(df))

    # 初始化线程池
    pool = multiprocessing.Pool(processes=thread_num)
    for _, row in tqdm(df.iterrows()):
        filename = row['filename']
        url = row['url']
        file = join(out_path, filename)
        # 如果已经存在，则跳过
        if os.path.exists(file):
            continue
        pool.apply_async(download, (file, url))  # 异步并行计算
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

