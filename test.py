from transformers import FlaubertForSequenceClassification, FlaubertTokenizer
import torch

# Create Model
model, log = FlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_large_cased', output_loading_info=True, problem_type = 'multi_label_classification', num_labels = 5)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_large_cased', do_lowercase=False)

# Tokenize
sentences = ["Le chat mange une pomme.", "Le chien court dans la rue.", "This is in English.", "hi"]
inputs = torch.tensor([flaubert_tokenizer.encode(sen, padding = 'max_length', max_length = 10) for sen in sentences])

print(inputs)

print(model.num_labels)

with torch.no_grad():
    logits = model(inputs).logits
    
pred = [log.argmax().item() for log in logits]

print(pred)
print(logits)