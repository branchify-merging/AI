import os
import os
import json
from transformers import T5Tokenizer

class SummaryDataLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir  # 여러 JSON 파일이 저장된 디렉토리 경로
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

    def load_data(self):
        inputs = []
        labels = []
        max_lens = []

        # 디렉토리 내 모든 JSON 파일을 처리
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # JSON이 딕셔너리 형식인지 확인
                if isinstance(data, dict):
                    passage = data['Meta(Refine)']['passage']
                    passage_cnt = data['Meta(Refine)']['passage_Cnt']
                    summary1 = data['Annotation']['summary1']
                    summary2 = data['Annotation']['summary2']

                    inputs.append(passage)
                    labels.append(f"{summary1} {summary2}")
                    max_lens.append(passage_cnt)

        return inputs, labels, max_lens


    def tokenize_data(self, inputs, labels, max_lens):
    
        inputs_tokenized = []
        labels_tokenized = []

        for input_text, label_text, max_len in zip(inputs, labels, max_lens):
            input_tokenized = self.tokenizer(
                input_text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            label_tokenized = self.tokenizer(
                label_text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            inputs_tokenized.append(input_tokenized)
            labels_tokenized.append(label_tokenized)

        return inputs_tokenized, labels_tokenized

    def create_batches(self, inputs_tokenized, labels_tokenized):
    
        for i in range(0, len(inputs_tokenized), self.batch_size):
            input_batch = inputs_tokenized[i:i + self.batch_size]
            label_batch = labels_tokenized[i:i + self.batch_size]
            yield input_batch, label_batch
