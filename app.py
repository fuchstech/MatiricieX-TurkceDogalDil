import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re
import gdown


app = FastAPI()


class Item(BaseModel):
    text: str = Field(..., example="Entity X'in müşteri hizmetleri hızlı ve etkili. Entity Y'nin ürün kalitesi çok kötü.")

# Model ve Tokenizer'ı yükleme
model_path = "./sentiment_model"
tokenizer_path = "./sentiment_tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Pipeline oluşturma
custom_sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
default_sentiment_analysis = pipeline("sentiment-analysis")  # Hazır model

#  Ek Entity listesi
entities_list = [
    "Superonline", "Paycell", "Lifecell", "Global", 
    "Kule Hizmet", "Superbox", "Enerji", "Teknoloji",
    "Sigorta", "Rehberlik ve Müşteri Hizmetleri", "BiP",
    "Yaani", "Platinum", "TV+", "Fizy", "Dergilik", "Akıllı Depo", "Hesabım", 
    "Dijital Operatör", "Gece Kuşu Tarifesi", "Mobil İmza", "Akademi", "Data Merkezi", 
    "Sağlık Metre", "Vodafone", "Twitch", "Kick", "Türk Telekom", "Superonline", 
    "D-Smart", "KabloTV", "Digitürk", "Türksat", "Bimcell", "Pttcell", "Vestel", 
    "Casper", "Arçelik", "Teknosa", "MediaMarkt", "Vatan Bilgisayar", "N11", 
    "Hepsiburada", "Trendyol", "Getir", "GittiGidiyor", "Sahibinden", "Letgo", 
    "Yemeksepeti", "Twitch", "YouTube", "Instagram", "Facebook", "Twitter", 
    "WhatsApp", "Snapchat", "BluTV", "Netflix", "Amazon Prime Video", "Exxen", 
    "Gain", "TikTok", "Xiaomi", "Oppo", "Huawei", "Samsung", "Apple", "Lenovo", 
    "HP", "Dell", "ASUS", "Microsoft", "Google", "X", "Entity X", "Entity Y"
]

def get_named_entities(text):
    entities = set()
    entities.update(re.findall(r'@\w+', text))
    entities.update(re.findall(r'\b(\w+?)\'\w+\b', text))

    words = text.split()
    for i, word in enumerate(words):
        if i == 0:
            continue
        if word in entities_list:
            entities.add(word)
        elif word[0].isupper():
            if i > 0 and words[i-1][-1] != '.':
                entities.add(word)
    
    return list(entities)

def split_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def simple_sentiment_analysis(text):
    positive_keywords = [
        "hızlı", "etkili", "iyi", "kaliteli", "güzel", "memnun",
        "mükemmel", "harika", "müthiş", "pozitif", "güvenilir", 
        "başarılı", "olağanüstü", "üstün", "tatmin edici"
    ]
    negative_keywords = [
        "kötü", "yavaş", "berbat", "kalitesiz", "şikayet", 
        "negatif", "korkunç", "iğrenç", "yetersiz", "hayal kırıklığı", 
        "rezalet", "sorunlu", "eksik", "saçma"
    ]

    for word in negative_keywords:
        if word in text:
            return "olumsuz"
    for word in positive_keywords:
        if word in text:
            return "olumlu"
    return "nötr"

def analyze_sentiment(sentence):
    # 1. Adım: Basit kural tabanlı analiz
    sentiment = simple_sentiment_analysis(sentence)
    if sentiment != "nötr":
        return sentiment

    # 2. Adım: Özel modelinizle analiz
    model_sentiment = custom_sentiment_analysis(sentence)
    model_sentiment_label = model_sentiment[0]['label'].lower()
    if model_sentiment_label == 'positive':
        return 'olumlu'
    elif model_sentiment_label == 'negative':
        return 'olumsuz'

    # 3. Adım: Hazır model ile analiz
    default_model_sentiment = default_sentiment_analysis(sentence)
    default_model_sentiment_label = default_model_sentiment[0]['label'].lower()
    if default_model_sentiment_label == 'positive':
        return 'olumlu'
    elif default_model_sentiment_label == 'negative':
        return 'olumsuz'

    # Eğer hâlâ nötr ise, en son nötr döndür
    return "nötr"

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    entities = get_named_entities(item.text)
    sentences = split_into_sentences(item.text)
    results = []

    for entity in entities:
        for sentence in sentences:
            if entity in sentence:
                final_sentiment = analyze_sentiment(sentence)
                results.append({
                    "entity": entity,
                    "sentiment": final_sentiment
                })
                break  # İlk uygun cümlede bulunca diğer cümlelere bakmayı bırakıyoruz
    
    result = {
        "entity_list": entities,
        "results": results
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200)
