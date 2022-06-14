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

import pandas as pd

class TweetLoader(Dataset):
    def __init__(self, path, transformer):
        dataset = pd.read_pickle('train_parser/dataset_df.pkl')
        dataset = dataset[['tweet', 'label']] #Remove author names

        self.data = dataset.to_dict('record')

        if transformer == 'flaubert':
            self.tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_large_cased', do_lowercase=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.tokenizer(data[idx]['tweet'])
        label = data[idx]['label']
        return {'tweet':text, 'label':label}