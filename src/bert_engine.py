import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Define SentimentModel class (same as before)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Extract values from current batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # argmax: when you have a multi-class classification task, where each input belongs to exactly one class.
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()            
            
            # sigmoid: when you have a multi-label classification task, where each input can belong to multiple classes or have multiple attributes.
            # preds = torch.sigmoid(logits)  # Apply sigmoid activation
            # preds = (preds > threshold).cpu().numpy().astype(int)  # Apply threshold


            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# def main():
#     config = load_config('config.json')

#     tokenizer = BertTokenizer.from_pretrained(config['model_name'])
#     train_dataset = SentimentDataset(
#         file_path=config['train_file'],
#         tokenizer=tokenizer,
#         label_mapping=config['label_mapping'],
#         max_sequence_length=config['max_sequence_length']
#     )
#     train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

#     model = SentimentModel(num_labels=len(config['label_mapping']), model_name=config['model_name'])
#     optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) // config['batch_size'] * config['num_epochs'])

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     for epoch in range(config['num_epochs']):
#         train_loss = train_epoch(model, train_dataloader, optimizer, device)
#         print(f"Epoch {epoch + 1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}")

#         # Evaluation
#         # Load test dataset and create test dataloader similar to train
#         # Call evaluate_model function and print accuracy
        
#         # Update scheduler
#         scheduler.step()

# if __name__ == '__main__':
#     main()
