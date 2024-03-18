import os
import time

import requests

import urllib.request
import urllib.error
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup


def images_download(search_words, images_num):

    # 데이터 수집
    s=requests.Session()        # 세션 생성
    for search_word in search_words:

        # 하위 디렉토리 생성
        subdir = 'images/' + str(search_word)       # 하위 디렉토리명 지정
        if not os.path.exists(subdir):      # 하위 디렉토리 없으면 생성
            os.makedirs(subdir)

        # 주소 접근
        url = "https://www.google.com/search?tbm=isch&q=" + str(search_word)
        response = s.get(url)       # 주소에 응답 요청
        print(response)
        
        soup = BeautifulSoup(response.text, 'html.parser')	 #soup 객체 생성
        time.sleep(5)
        imgs = soup.find_all("img")       # 모든 img 선택
        print(len(imgs))

        num = 0
        for img in imgs:
            src = img['src']
            try:
                urllib.request.urlretrieve(src, f'{subdir}/{str(num)}.jpg')    # 이미지 디렉토리에 저장
                num += 1
            except:
                print('이미지 오류')

            if num == images_num:       # images_num개 만큼 저장하면 종료
                break

    s.close()       # 세션 종료

search_words = ["벤치프레스", "데드리프트", "스쿼트"]
images_num = 40
images_download(search_words, images_num)