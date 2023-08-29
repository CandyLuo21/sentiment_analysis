import pandas as pd
import torch
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_mapping, max_sequence_length):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.label_mapping[self.data.iloc[idx]['sentiment']]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# def load_config(config_path):
#     with open(config_path, 'r') as config_file:
#         config = json.load(config_file)
#     return config

# if __name__ == '__main__':
#     config = load_config('config.json')

#     tokenizer = BertTokenizer.from_pretrained(config['model_name'])
#     dataset = SentimentDataset(
#         file_path=config['train_file'],
#         tokenizer=tokenizer,
#         label_mapping=config['label_mapping'],
#         max_sequence_length=config['max_sequence_length']
#     )

    # Now you can use the dataset for training
    # ... (rest of the training loop)
