import os
import time

import requests

import urllib.request
import urllib.error
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup


# 이미지 저장
def images_download(search_words, image_types, images_num):

    s=requests.Session()        # 세션 생성
    for search_word in search_words:        # 단어 설정

        # 하위 디렉토리 생성
        subdir = 'images/' + str(search_word)       # 하위 디렉토리명 지정
        if not os.path.exists(subdir):      # 하위 디렉토리 없으면 생성
            os.makedirs(subdir)

        for i, image_type in enumerate(image_types):    # 이미지 타입 설정
            url = f"https://www.google.com/search?tbm=isch&tbs={image_type}&q={search_word}"    # url 설정
            response = s.get(url)       # 주소에 응답 요청
            print(response)
            
            soup = BeautifulSoup(response.text, 'html.parser')	 #soup 객체 생성
            time.sleep(3)

            imgs = soup.find_all("img")       # 모든 img 선택
            print(search_word, image_type, len(imgs))

            img_num = 0
            for img in imgs:
                src = img['src']        # 이미지 주소 받기
                try:
                    urllib.request.urlretrieve(src, f'{subdir}/{str(img_num + (i * images_num))}.jpg')    # 이미지명 설정하여 디렉토리에 저장
                    img_num += 1
                except:
                    print('이미지 오류')

                if img_num == images_num:       # images_num개 만큼 저장하면 종료
                    break

    s.close()       # 세션 종료

search_words = ["벤치프레스", "데드리프트", "스쿼트", "오버헤드프레스", "풀업"]
image_types = ['itp:lineart,ic:gray', 'itp:animated,ic:gray', 'itp:animated,ic:color', 'itp:photo,ic:gray', 'itp:photo,ic:color', 'itp:photo,ic:trans']
images_num = 20
images_download(search_words, image_types, images_num)