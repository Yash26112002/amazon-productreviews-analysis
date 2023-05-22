from fastapi import FastAPI
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import apiextract
from flair.data import Sentence
from flair.models import TextClassifier
import backend_ml
import os

app = FastAPI()
sia = TextClassifier.load('en-sentiment')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specify the origins that are allowed to make requests (update as needed)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    label = sentence.labels[0]
    if label.value == "POSITIVE":
        return 1 
    elif label.value == "NEGATIVE":
        return -1
    else:
        return 0

def process_csv(file_path):
    # Read the CSV file
    df1 = pd.read_csv(file_path)
    df=df1[['Rating','Title','Description']]
    df['Reviews'] = df['Title'] + ' . ' + df['Description']
    df = df.drop(['Title','Description'], axis=1)
    df=backend_ml.preprocess(df)
    df["sentiment_score"] = 0
    for i, row in df.iterrows():
        sentiment_score = flair_prediction(row["Reviews"])
        df.at[i, "sentiment_score"] = sentiment_score
    sent_cnt=backend_ml.sentiment_count(df)
    top_pos=backend_ml.get_top_positive_words(df)
    top_neg=backend_ml.get_top_negative_words(df)
    # print(sent_cnt)
    # print(top_pos)
    # print(top_neg)
    if os.path.exists(file_path):  # Check if the file exists
        os.remove(file_path)
    return [
        {'sent_cnt': sent_cnt},
        {'top_pos': top_pos},
        {'top_neg': top_neg}
        
    ]
    
    
    
@app.post("/process_link")
async def process_link(link: dict):
    print('succesful1')
    link_value = link.get('link')
    apiextract.main(link_value)
    print('succesful')
    return {"message": "Link received and processed successfully!"}

@app.get("/process-csv")
async def process_csv_endpoint():
    file_path = "C:/Users/YASH SANGWAN/Desktop/dst project/output.csv" 
    # print("hfjow") 
    sent_cnt, top_pos, top_neg  = process_csv(file_path)
    return {'sent_cnt': sent_cnt,'top_pos': top_pos,'top_neg': top_neg}
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)