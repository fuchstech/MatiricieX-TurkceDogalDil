#written by HIZMALI
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Excel dosyasını yüklemek için pandas'ın read_excel fonksiyonunu kullanın
df = pd.read_excel(r'C:\Users\dest4\Desktop\hzmlm\lp_all.xlsx')


# Veri Hazırlama
labels = df.iloc[:, 0].tolist()  # A sütunundaki etiketler (-1, 1)
texts = df.iloc[:, 1].tolist()   # B sütunundaki cümleler

# Etiketleri 0 ve 1'e dönüştürme (BERT modeline uygun hale getirmek için)
labels = [0 if label == -1 else 1 for label in labels]

# Veriyi eğitim ve doğrulama setlerine ayırma
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# BERT tokenizer kullanarak metinleri tokenize etme
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Dataset sınıfını oluşturma
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Eğitim için gerekli argümanlar
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Eğitim süresini artırabilirsiniz
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_total_limit=2,  # Maksimum 2 kontrol noktası sakla
    save_steps=500,  # Her 500 adımda bir modeli kaydet
)

# Modeli yükleme
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 2 sınıf (negative, positive)

# Eğitici (Trainer) oluşturma
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Modeli eğitme
trainer.train()

# Modeli Kaydetme
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_tokenizer')

# Değerlendirme sonuçları
eval_result = trainer.evaluate()
print(eval_result)
