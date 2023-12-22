import os
from torch import nn, Tensor
import torch
import logging

from typing import Optional
from sentence_transformers import InputExample
from transformers import DataCollator, DataCollatorForWholeWordMask, DataCollatorWithPadding
from transformers import Trainer
from dataclasses import dataclass
from transformers import AutoModel
import torch.distributed as dist
from transformers.file_utils import ModelOutput
import datasets

class YottaDatasetForTraining(torch.utils.data.Dataset):
    
    def _load_data(self, path):
        return datasets.load_dataset('json', data_files= path, split='train')
    
    def __init__(self, dataset_paths : List[str]):
        self.dataset_paths = dataset_paths
        datasets_data = []
        for dataset in dataset_paths:
            tmp_data = self._load_data(dataset)
            datasets_data.append(tmp_data)
            print(dataset)
            
        self.data = datasets.concatenate_datasets(datasets_data)
        print('duzina svega: ', len(self.data))
        self.total_len = len(self.data)
    
    
    def __getitem__(self, item):
        query = self.data[item]['query']
        pos = self.data[item]['pos']
        return InputExample(texts=[query, pos])

    def __len__(self):
        return self.total_len


class YottaCollator(DataCollatorForWholeWordMask):
    
    def __call__(self, examples):

        num_texts = len(examples[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in examples:
            for idx, text in enumerate(example.texts):
                texts[idx].append(str(text))

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenizer(texts[idx], padding=True, 
                                  truncation='longest_first',
                                  return_tensors="pt", max_length=128)
            sentence_features.append(tokenized)

        return sentence_features, labels
    
@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    
    query_max_len: int = 64
    passage_max_len: int = 150

    def __call__(self, examples):
        query = [f.texts[0] for f in examples]
        passage = [f.texts[1] for f in examples]

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}

from sentence_transformers import SentenceTransformer
import sentence_transformers.models as models
def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)
    
    
logger = logging.getLogger(__name__)
class YottaTrainer(Trainer):
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode='mean')

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss



@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class YottaEmbedder(nn.Module):

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            
            scores = self.compute_similarity(q_reps, p_reps)
            scores_q = self.compute_similarity(q_reps, q_reps)
            scores_q[torch.eye(scores_q.shape[0], dtype=torch.bool)] = 0.0
            scores_p = self.compute_similarity(p_reps, p_reps)
            scores_p[torch.eye(scores_q.shape[0], dtype=torch.bool)] = 0.0
            
            scores_1 = torch.cat((scores, scores_q), dim=1)
            scores_2 = torch.cat((scores_p, scores.transpose(0,1)), dim=1)
            scores = torch.cat((scores_1, scores_2), dim=0)
            scores = scores / self.temperature
            
            # TODO enable triplet loss
            # scores = scores.view(q_reps.size(0)*2, -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            # target = target * (p_reps.size(0) // q_reps.size(0))

            ## logging
            max_ids = torch.max(scores, 1)[1]
            logger.info(torch.sum(max_ids != target))
             
            loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
