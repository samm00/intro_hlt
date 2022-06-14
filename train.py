# Minimal training procedure for a pytorch model

# Constants/hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = .0001
EPOCHS = 10
DISPLAY = 25
chkpt_path = '/saltpool0/scratch/layneberry/CLIPC/spokencoco_aligned_with_frequencies.pth'
load_chkpt = False
workers = 50
transformer = 'None'

# Step 0: Import everything 
import torch as th
from torch.utils.data import DataLoader
from dataloader import TweetLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

# Step 1: Create an instance of the model
from model import Model
model = Model(transformer)
model = th.nn.DataParallel(model)
if load_chkpt:
    model.load_state_dict(th.load(chkpt_path))

# Step 2: Create an instance of each dataset (train, val, test)
train_dataset = DataLoader(TweetLoader('train.tsv',transformer),batch_size=BATCH_SIZE,shuffle=True,num_workers=workers,drop_last=True)
val_dataset = DataLoader(TweetLoader('val.tsv',transformer),batch_size=BATCH_SIZE,shuffle=True,num_workers=workers,drop_last=True)

# Step 3: Create instances of loss function and optimizer
loss_op = CrossEntropyLoss()
evaluate = TODO: Accuracy
opt = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 4: Training loop over epochs, then over dataset
model.to(device)

itr = 0
running_loss = 0.0
for e in range(EPOCHS):
    print('Epoch', e)
    model.train()
    for batch in tqdm(train_dataset):
        opt.zero_grad()
        with th.set_grad_enabled(True):
            # Step 5: In inner loop, apply model to batch
            predictions = model(batch['tweet']) 
            
            # Step 6: In inner loop, compute loss based on batch
            loss = loss_op(predictions, batch['label'])

            # Step 7: In inner loop, backprop+step
            loss.backward()
            opt.step()
            
            # Step 8: In inner loop, if at display interval, print loss
            if itr % DISPLAY == 0:
                print("Training loss at iteration", itr, running_loss / DISPLAY)
                running_loss = 0.0 
            itr += 1

    # Step 9: In outer loop, compute metrics on val set and report
    evaluate(model,val_dataset)
    if chkpt_path:
        th.save(model.state_dict(), chkpt_path)
