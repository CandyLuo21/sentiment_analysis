import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import json
import time
from bert_dataset import BertDataset
from bert_model import BertModel
from bert_engine import *


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def train_model():
    # Record the start time
    start_time = time.time()  
    config = load_config('sentiment_analysis/src/config.json')

    tokenizer = BertTokenizer.from_pretrained(config['model_name'], do_lower_case=True)
    train_dataset = BertDataset(
        file_path=config['train_file'],
        tokenizer=tokenizer,
        label_mapping=config['label_mapping'],
        max_sequence_length=config['max_sequence_length']
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = BertModel(num_labels=len(config['label_mapping']), model_name=config['model_name'], dropout = config['dropout'])
    
    # Exclude weight decay for bias, LayerNorm.bias, and LayerNorm.weight
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

    
    # learning rate scheduler 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) // config['batch_size'] * config['num_epochs'])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(device)
   
    early_stopping_patience = 3  # Number of epochs to wait before early stopping
    early_stopping_counter = 0  # Counter to track patience
    
    # Initialize best_accuracy to track the best accuracy achieved
    best_accuracy = 0.0  
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}")        

        # Evaluation
        # Load test dataset and create test dataloader similar to train
        test_dataset = BertDataset(
            file_path=config['test_file'],
            tokenizer=tokenizer,
            label_mapping=config['label_mapping'],
            max_sequence_length=config['max_sequence_length']
        )
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        accuracy = evaluate_model(model, test_dataloader, device)        
        epoch_training_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{config['num_epochs']}: Test Accuracy = {accuracy:.4f} using {epoch_training_time:.2f} seconds")

        # Save the model if the current accuracy is the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # # Save the model's state dictionary
            torch.save(model.state_dict(), config['output_dir'] + 'bert_model.bin')            
            early_stopping_counter = 0  # Reset early stopping counter
        else:
            early_stopping_counter += 1
        
        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Stopping training.")
            break
            
        # Update scheduler
        scheduler.step()
    
    # Record the end time    
    end_time = time.time()
    # Calculate the training time in seconds 
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")   

if __name__ == '__main__':
    train_model()
