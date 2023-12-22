from torch.utils.data import Dataset
from typing import List
from sentence_transformers import InputExample
import numpy as np
import random
import logging

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

logger = logging.getLogger(__name__)

class TypingDataset(Dataset):
    """
    The TypingDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """
    def __init__(self, sentences: List[str], typing_errors_filepath : str):
        self.sentences = sentences
        # self.noise_fn = noise_fn
        self.typing_errors_filepath = typing_errors_filepath
        self.cnt = 0
        self.typing_cnt = 0
        self.aug = naw.SpellingAug(aug_max=3)
        with open(typing_errors_filepath, newline='\n') as rf:
            lines = rf.readlines()
            
            lines = [line[:-1].split('\t') for line in lines]
            
            self.typing_dict: dict[str,str] = {str(line[1]) : str(line[0]) for line in  lines}


    def __getitem__(self, item):
        sent : str = self.sentences[item]
        return InputExample(texts=[self.create_typo(sent), sent], label=1.0)


    def __len__(self):
        return len(self.sentences)

    
    # def create_typo(self, text: str, typo_ratio=0.2):
    #     words = str(text).split(' ')
        
    #     ids = [idx for idx, word in enumerate(words) if word in self.typing_dict]
    #     self.cnt+=1
        
    #     # if self.cnt%10000 == 0:
    #     #     logger.info(f'procentage of typos added {self.typing_cnt/self.cnt*100:.4f}%')
        
    #     if len(ids)==0:
    #         return text
    #     self.typing_cnt+=1
        
    #     id = random.choice(ids)
    #     words[id] = self.typing_dict[words[id]]
    #     new_text = ' '.join(words)
    #     return new_text
    
    def create_typo(self, text: str, typo_ratio=0.2):
        
        
        new_text = self.aug.augment(str(text), n=1)[0]
        # print(new_text)
        return new_text
    
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict

class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
    is used when extending sentence embeddings to new languages as described in our publication
    Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

    For an example, see the documentation on extending language models to new languages.
    """
    def __init__(self, model, scale: float = 20.0):
        """
        :param model: SentenceTransformerModel
        """
        super(MSELoss, self).__init__()
        self.model = model
        self.scale = scale
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        # print(embeddings_a.shape, embeddings_b.shape)
        
        scores = (1 - torch.einsum('ij,ij->i', embeddings_a, embeddings_b) ) * self.scale
        labels = torch.zeros(len(scores), dtype=torch.float, device=scores.device)  # Example a[i] should match with b[i]
        return self.loss_fct(scores, labels), None, None
