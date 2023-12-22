## imports
import os
import deepspeed
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import logging
import json
logging.basicConfig(filename='./logs/train-sum-qa-deepspeed-test.log', level=logging.INFO)
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, models, util, evaluation, losses, InputExample
from TypingDataset import TypingDataset, MSELoss

## data_paths for NLI

data_folder_path = '/drive02/novak/datasets/downloads/qa/'

data_filenames = [
    # 'gooaq_pairs.jsonl',
    'yahoo_answers_title_answer.jsonl',
    # 'stackexchange_duplicate_questions_title_title.jsonl',
    # 'eli5_question_answer.jsonl',
    # 'yahoo_answers_title_question.jsonl',
    # 'yahoo_answers_question_answer.jsonl',
    # 'wikihow.jsonl',
    # 'NQ-train_pairs.jsonl',
    # 'amazon-qa-train-pairs.jsonl',
    # 'squad_pairs.jsonl',
    # 'quora_duplicates.jsonl',
    # 'PAQ_pairs.jsonl',
]

def load_data(filename):
    
    ret = []
    with open(f'{data_folder_path}{filename}', 'r') as rf:
        
        # lines = rf.readlines()
        
        lines = [InputExample(texts=json.loads(line)) for line in tqdm(rf)]
        
        return lines
        
        # for line in tqdm(lines):
        #     line_data = InputExample(texts=json.loads(line))
        #     ret.append(line_data)
        
    # return ret
            
data = []
for dataset in data_filenames:
    
    data += load_data(dataset)
    
    
print(len(data))
pair_data_loader = DataLoader(data, batch_size=300*2, shuffle=True)
# pair_data_loader = TypingDataLoader(data, batch_size=300*4, shuffle=True)


model_name = 'all-MiniLM-L12-v2'
print('djovak')
model = SentenceTransformer(model_name)

tokenizer_args = {
  "clean_up_tokenization_spaces": True,
  "cls_token": "[CLS]",
  "do_basic_tokenize": True,
  "do_lower_case": True,
  "mask_token": "[MASK]",
  "max_length": 128,
  "model_max_length": 512,
  "never_split": None,
  "pad_to_multiple_of": None,
  "pad_token": "[PAD]",
  "pad_token_type_id": 0,
  "padding_side": "right",
  "sep_token": "[SEP]",
  "stride": 0,
  "strip_accents": None,
  "tokenize_chinese_chars": True,
  "tokenizer_class": "BertTokenizer",
  "truncation_side": "right",
  "truncation_strategy": "longest_first",
  "unk_token": "[UNK]"
}


import torch
model_name = 'qa-yotta-L12-v1'
word_embedding_model = models.Transformer('microsoft/MiniLM-L12-H384-uncased', max_seq_length=512, tokenizer_args=tokenizer_args)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=384, activation_function=torch.nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])


negative_loss = losses.MultipleNegativesRankingLoss(model=model, name='negative', similarity_fct=util.cos_sim, scale=80)

import os
local_rank = int(os.environ['LOCAL_RANK'])

# import os
# print(os.environ['LOCAL_RANK'])

ds_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.0002,
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_num_steps": 200
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False,
        "number_checkpoints": 10,
        "synchronize_checkpoint_boundary": False,
        "profile": True
    },
    
#     "zero_optimization": {
#         "stage": 3,
#         "contiguous_gradients": True,
#         "stage3_max_live_parameters": 1e9,
#         "stage3_max_reuse_distance": 1e9,
#         "stage3_prefetch_bucket_size": 1e7,
#         "stage3_param_persistence_threshold": 1e5,
#         "reduce_bucket_size": 1e7,
#         "sub_group_size": 1e9,
#         "offload_optimizer": {
#             "device": "cpu"
#          },
#         "offload_param": {
#             "device": "cpu"
#        }
#    },
    
    # "zero_optimization": {
    #     "stage": 1,
    #     "reduce_bucket_size": 5e8
    # },
    
    "comms_logger": {
        "enabled": True,
        "verbose": False,
        "prof_all": True,
        "debug": False
    },

    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 500
}

engine, optimizer, eng_dataloader, *_ = deepspeed.initialize(
    model=negative_loss,
    config= ds_config,
    training_data=data,
    collate_fn=model.smart_batching_collate,
    
    
)

from tqdm import tqdm
from sentence_transformers.util import batch_to_device
import os
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '9994'
# os.environ['RANK'] = "0"
# os.environ['LOCAL_RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"

# if args.local_rank == -1: 
#     args.local_rank = 0

# 3. finally init deepspeed dist and set the default device
# deepspeed.init_distributed()
# device = torch.device("cuda", 0)

# pair_data_loader.collate_fn = model.smart_batching_collate

for idx, batch in tqdm(enumerate(eng_dataloader)):
    

    features, labels = batch
    labels.to(device=local_rank)
    features = list(map(lambda batch: batch_to_device(batch, local_rank), features))
    loss = engine(features, labels)
    engine.backward(loss)
    engine.step()
    
    
    
    
    if idx%10 == 0:
    
        logging.info(f'loss on step {idx}:{loss.item()}')
        
    if idx%200 == 0:
        os.makedirs(f'./test-real/yotta-embeder-v2/{idx}', exist_ok=True)
        model.save(f'./test-real/yotta-embeder-v2/{idx}')
        print(f'saved model {idx}')