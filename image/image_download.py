from lib2to3.pgen2 import driver
from turtle import down
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import urllib.request
import time

img_class_name=".rg_i.Q4LuWd" 


# keyword = input("keyword:")
# limit = int(input("limit :"))

# limit = 10
def browser():
    chromedriver_path='./chromedriver'

    #option setting
    options = Options()
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.70 Whale/3.13.131.27 Safari/537.36"
    options.add_argument('user-agent=' + user_agent)
    options.add_argument('headless')
    options.add_argument('--start-fullscreen')
    # options.add_argument("window-size=1920x1080")
    options.add_argument('--blink-settings=imagesEnabled=false') 
    options.add_argument("disable-gpu") 
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")

    ### driver
    driver = webdriver.Chrome(chromedriver_path,options=options)

    return driver

def scroll_down(driver,scroll_history):
    """
    구글 이미지 검색에서 스크롤을 내려서 로딩을 해주어야한다. 
    반복문을 사용해서 전체 이미지 로딩 해주기 
    """
    while True:
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        now = driver.execute_script('return document.body.scrollHeight') 
        scroll_history.append(now)
        if(scroll_history[-1]==scroll_history[-3]):
            try:
                driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()
            except:
                break

def move_base_url(driver):
    base_url ='https://images.google.com/'
    driver.get(base_url)
    return driver

def download(driver,keyword):
    move_base_url(driver) 
    # 파일 이름 받아오기 
    file_name = keyword.replace(' ','_')
    # 검색창 입력 
    keyword = keyword+"\n"
    search_input_path = '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input'
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.70 Whale/3.13.131.27 Safari/537.36"
    
    driver.find_element_by_xpath(search_input_path).send_keys(keyword)

    scroll_history = [0,driver.execute_script('return document.body.scrollHeight')]
            
    scroll_down(driver,scroll_history=scroll_history)
    source = driver.page_source
    soup = BeautifulSoup(source,"html.parser")
    img = soup.select(img_class_name)
    

    for idx,image in enumerate(img):
        # print("IMAGE",idx)
        opner = urllib.request.build_opener()
        opner.addheaders=[('User-Agent',user_agent)]
        try:
            urllib.request.install_opener(opner)
            urllib.request.urlretrieve(image.attrs["src"],"./data/oneline"+str(idx)+".jpg")
            # temp_img.append(image.attrs["src"])
        except KeyError:
            try:
                urllib.request.install_opener(opner)
                urllib.request.urlretrieve(image.attrs["data-src"],"./data/"+str(file_name)+str(idx)+".jpg")
            except:
                pass
    

keyword =["one line drawing","one line drawing face","one line artwork"]

driver = browser()

for w in keyword:
    download(driver,w)

driver.close