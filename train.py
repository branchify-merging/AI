import os
import torch
from data_loader.data_loader import SummaryDataLoader
from model.model import CommitMessageModel
from trainer.trainer import Trainer

# 데이터 로더 
data_loader = SummaryDataLoader(data_dir="data/1.Training/train_labled/TL1/05.minute/2~3sent", batch_size=32)
inputs, labels, max_lens = data_loader.load_data()
inputs_tokenized, labels_tokenized = data_loader.tokenize_data(inputs, labels, max_lens)

model = CommitMessageModel()
trainer = Trainer(model=model, data_loader=data_loader.create_batches(inputs_tokenized, labels_tokenized))

trainer.train()

def save_model(model, save_dir='saved/models/', model_name='commit-analysis_model.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    
    model_path = os.path.join(save_dir, model_name)
    torch.save(model.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

save_model(model)