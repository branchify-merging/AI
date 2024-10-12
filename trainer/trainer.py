import torch
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, data_loader, learning_rate=5e-5, num_epochs=3):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        self.model.train()  # 모델을 학습 모드로 전환
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            epoch_loss = 0
            for input_batch, label_batch in tqdm(self.data_loader):
                # input_batch가 리스트인 경우 처리
                if isinstance(input_batch, list):
                    input_batch = input_batch[0]

                if isinstance(label_batch, list):
                    label_batch = label_batch[0]
                
                # 딕셔너리로 변환된 후 device로 전송
                input_batch = {k: v.to(self.device) for k, v in input_batch.items()}
                label_batch = {k: v.to(self.device) for k, v in label_batch.items()}

                # 손실 계산
                outputs = self.model.forward(input_batch['input_ids'], input_batch['attention_mask'], label_batch['input_ids'])
                loss = outputs.loss

                # 옵티마이저
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f"Loss: {epoch_loss / len(self.data_loader)}")


    def evaluate(self):
        self.model.eval()  # 모델을 평가 모드로 전환
        with torch.no_grad():
            for input_batch, label_batch in tqdm(self.data_loader):
                input_batch = {k: v.to(self.device) for k, v in input_batch.items()}
                label_batch = {k: v.to(self.device) for k, v in label_batch.items()}

                outputs = self.model.generate(input_batch['input_ids'], input_batch['attention_mask'], max_length=50)
            
