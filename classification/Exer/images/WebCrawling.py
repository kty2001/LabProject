from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import os
import time

import urllib.request
import urllib.error
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup


# 이미지 저장
def images_download(search_word, images_num):
    driver.get(f"https://www.google.com/imghp?hl=ko&tab=wi")
    for search_word in search_words:        # 단어 설정
        url = None
        time.sleep(5)
        # 하위 디렉토리 생성
        subdir = './' + str(search_word)       # 하위 디렉토리명 지정
        if not os.path.exists(subdir):      # 하위 디렉토리 없으면 생성
            os.makedirs(subdir)

        elem = driver.find_element(By.NAME, "q")
        elem.clear()
        elem.send_keys(search_word)
        elem.send_keys(Keys.RETURN)
        time.sleep(3)

        # 스크롤 다운
        SCROLL_PAUSE_TIME = 1.5
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scrall height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_element(By.XPATH, "//input[@value='결과 더보기']").click()
                except:
                    break
            last_height = new_height

        # beautifulsoup로 페이지 html 읽기
        url = driver.page_source
        time.sleep(3)

        soup = BeautifulSoup(url, 'html.parser')	 #soup 객체 생성
        time.sleep(3)

        imgs = soup.find_all("img", class_="rg_i Q4LuWd")       # 모든 img 선택
        print(search_word, len(imgs))

        img_num = 0
        for img in imgs:
            try:
                src = img['src']        # 이미지 주소 받기
                urllib.request.urlretrieve(src, f'{subdir}/{str(img_num)}.jpg')    # 이미지명 설정하여 디렉토리에 저장
                img_num += 1
                if img_num % 10 == 0:
                    print("now: ", img_num)
                    time.sleep(1)
            except:
                print('이미지 오류')

            if img_num == images_num:       # images_num개 만큼 저장하면 종료
                break
    
    driver.quit()

options = Options()
options.add_argument('--disable-blink-features=AutomationControlled')

search_words = ["pullup", "plank", "squat", "deadlift", "benchpress"]
images_num = 150

chromedriver_path = './chromedriver.exe'
driver = webdriver.Chrome(executable_path=chromedriver_path)


images_download(search_words, images_num)
