import requests
import pandas as pd
import random
import math
from bs4 import BeautifulSoup
headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.amazon.in/",
        "DNT": "1",
        "Connection": "keep-alive"
}
review_list=[]
def extractreviews(review_url):

    resp = requests.get(review_url, headers=headers)
    # print(resp)
    soup=BeautifulSoup(resp.text,'html.parser')
    reviews=soup.findAll('div',{'data-hook':"review"})
    # print(reviews)
    # print(len(reviews))
    t=0
    for item in reviews:
        with open('outputs/file.html', 'w', encoding='utf-8') as f:
            f.write(str(item))
           
        
        review={
            "reviews.username": item.find('span',{'class':"a-profile-name"}).text.strip(),
            "Rating":int(item.find('i',{"data-hook":"review-star-rating"}).text.strip()[0]),
            "Title":item.find('a', {"data-hook":"review-title"}).text.strip(),
            "reviews.date":item.find('span',{"data-hook":"review-date"}).text.strip(),
            "Description":item.find('span', {'data-hook': 'review-body'}).text.strip()
        }
        review_list.append(review)
        
def totalpages(review_url):
    resp = requests.get(review_url, headers=headers)
    soup=BeautifulSoup(resp.text,'html.parser')
    reviews=soup.find('div',{'data-hook':"cr-filter-info-review-rating-count"})
    pg=(int(reviews.text.strip().split(' ')[3].replace(',','')))
    pg=math.ceil(pg/10)
    return(pg)
    
    
def main(url):
    # url="https://www.amazon.in/Noise-ColorFit-Display-Monitoring-Smartwatches/dp/B09NVPDLNV/ref=sr_1_5?crid=2ZLVN1LSG3GAU&keywords=watch&qid=1684341668&sprefix=watch%2Caps%2C482&sr=8-5&th=1"
    s=""
    for i in url:
        if(i=='?'):
            break
        s+=i
    url=s
    # pgnumber=2
    # print(reviews_url)
    reviews_url=url.replace("dp","product-reviews")+"?pageNumber="+str(1)
    totpgs=totalpages(reviews_url)
    for pagenum in range(1,totpgs+1):
        # print(f"running for page {pagenum}")
        try:
            reviews_url=url.replace("dp","product-reviews")+"?pageNumber="+str(pagenum)
            extractreviews(reviews_url)
        except Exception as e:
            print(e)
    df=pd.DataFrame(review_list)
    df.to_csv('output.csv',index=False)
    
    

