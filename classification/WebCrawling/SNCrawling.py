from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import time

# ChromeDriver 경로 설정
# chromedriver_path = '/home/kty/LabProject/classification/WebCrawling/chromedriver'

# Chrome WebDriver를 사용하여 브라우저 시작
driver = webdriver.Chrome()

# Google 검색 페이지 열기
driver.get("https://www.google.com")

# 검색어 입력
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("OpenAI")
search_box.send_keys(Keys.RETURN)

# 검색 실행
search_box.submit()

# 페이지가 로드될 때까지 잠시 대기
time.sleep(3)

# 첫 번째 검색 결과의 링크 클릭
first_link = driver.find_element(By.CSS_SELECTOR, "div.tF2Cxc a")
first_link.click()

# 페이지 타이틀 출력
print(driver.title)

# 브라우저 종료
driver.quit()