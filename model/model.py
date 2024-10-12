from transformers import T5ForConditionalGeneration
import torch
import torch.nn as nn

class CommitMessageModel(nn.Module):  
    def __init__(self, model_name='google/mt5-small'):
        super(CommitMessageModel, self).__init__()  
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def generate(self, input_ids, attention_mask, max_length=50): 
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
