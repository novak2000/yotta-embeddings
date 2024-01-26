## imports
import os
import torch

from transformers import AutoTokenizer, TrainingArguments, HfArgumentParser
from custom_models import FinetuneCollator, YottaTrainer, YottaDatasetForFinetuning, YottaCollator, YottaFinetuneEmbedder
import logging


os.environ["WANDB_PROJECT"] = "yotta-embedings-test" # name your W&B project 
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints

# DONE:
# vizulizuj duzinu query/passage u tokenima, da vidimo koliko gubimo njihovim ogranicenjem
# vizualizacija loss-a
## dodaj ostale kverije kao negativne
## proveri learning rate/weight decay

# TODO automatska evaluacija preko mtab na svakih N epoha, [MOZDA]rezultati ce biti vraceni u json-u pa ih upisi u neki jednostavniji format negde
# TODO teorija, ono sto ga cini losijim je seckanje passage-a, probaj da podelis set tako da izdvojs sve krace od 128 za pretraining, onda finetune na >128 dataset-u


import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)

logger = logging.getLogger(__name__)

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    
    ## training args
    parser = HfArgumentParser((RetrieverTrainingArguments, ))
    training_args = parser.parse_args_into_dataclasses()[0]
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)
    
    
    ## model
    model = YottaFinetuneEmbedder('./yotta-test-128-bigdata2-loss2.0/checkpoint-16200/', 
                          normlized=True, 
                          sentence_pooling_method='mean', 
                          negatives_cross_device=True, 
                          temperature=training_args.temperature)
    print(torch.cuda.device_count())


    ## datasets
    data_folder_path = '/drive02/novak/datasets/qa_triplets/out/'
    data_filenames = [
        'AllNLI.jsonl',
        'msmarco-triplets.jsonl',
        'quora_duplicates_triplets.jsonl',
        'specter_train_triples.jsonl'
    ]

    dataset_paths = [f'{data_folder_path}{data_filename}' for data_filename in data_filenames]
    dataset = YottaDatasetForFinetuning(dataset_paths)

    ## tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./yotta-test-128-bigdata2-loss2.0/checkpoint-16200/')

    ## data_collator
    data_collator = FinetuneCollator(tokenizer)


    ## trainer
    trainer = YottaTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
    )

    trainer.train()
    
if __name__ == '__main__':  
    main()