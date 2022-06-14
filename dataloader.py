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
    def __init__(self, path, transformer):
        # CBeaune.txt  CZacharopoulou.txt  dataloader.py  franckriester.txt  JLMelenchon.txt  MinColonna.txt 
        self.data = pd.read_csv(path, sep='\t')
        if transformer == 'flaubert':
            self.tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_large_cased', do_lowercase=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.tokenizer(data[idx]['tweet'])
        label = data[idx]['label']
        return {'tweet':text, 'label':label}