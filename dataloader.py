from torch.utils.data import Dataset
import json
import pickle as pkl
from PIL import Image
import soundfile as sf
import numpy as np
from CLIP import clip
from transformers import Wav2Vec2Processor, HubertModel
import torch as th
import time

class TweetLoader(Dataset):
    def __init__(self):
       # CBeaune.txt  CZacharopoulou.txt  dataloader.py  franckriester.txt  JLMelenchon.txt  MinColonna.txt 
       self.data = pd.read_csv('train_clean.tsv', sep='\t')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = data[idx]['tweet']
        label = data[idx]['label']
        return {'tweet':text, 'label':label}