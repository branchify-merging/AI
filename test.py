from data_loader.data_loader import SummaryDataLoader
from model.model import CommitMessageModel
from trainer.trainer import Trainer
import torch
import os

def load_trained_model(model_path='saved/models/commit-analysis_model.pth'):
    model = CommitMessageModel()
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.model.eval()  # 평가 모드로 전환
    return model

# 테스트 데이터 로더
test_loader = SummaryDataLoader(data_dir="data/2.Vaildation/val_labled/VL1/05.minute/2~3sent", batch_size=32)
inputs, labels, max_lens = test_loader.load_data()
inputs_tokenized, labels_tokenized = test_loader.tokenize_data(inputs, labels, max_lens)

# 모델 로드
model = load_trained_model()

# 트레이너 설정 및 평가
trainer = Trainer(model=model, data_loader=None)
trainer.evaluate(test_loader.create_batches(inputs_tokenized, labels_tokenized))