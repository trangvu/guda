'''
Created by trangvu on 10/12/20
Adapt from https://github.com/roeeaharoni/unsupervised-domain-clusters
'''
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


from collections import Counter, defaultdict

from transformers import *
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

import time
import os

MODELS = [
  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-multilingual-cased'),
  (RobertaModel, RobertaTokenizer, 'roberta-base'),
  (RobertaModel, RobertaTokenizer, 'roberta-large'),
  (BertModel, BertTokenizer, 'bert-base-uncased'),
  (BertModel, BertTokenizer, 'bert-large-cased'),
  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
  (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
  (GPT2Model, GPT2Tokenizer, 'gpt2'),
  (CTRLModel, CTRLTokenizer, 'ctrl'),
  (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
  (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
  (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
  (BertModel, BertTokenizer, 'bert-base-multilingual-cased'),
  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-large')
]


def encode_with_transformers(corpus, models_to_use=['distilbert-base-multilingual-cased']):
  """
  Encodes the corpus using the models in models_to_use.
  Returns a dictionary from a model name to a list of the encoded sentences and their encodings.
  The encodings are calculatd by average-pooling the last hidden states for each token.
  """
  model_to_states = {}
  for model_class, tokenizer_class, model_name in MODELS:
    if model_name not in models_to_use:
      continue
    logger.info('encoding with {}...'.format(model_name))
    model_to_states[model_name] = {}
    model_to_states[model_name]['states'] = []
    model_to_states[model_name]['sents'] = []

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.to(torch.device('cuda'))

    # Encode text
    start = time.time()
    for sentence in corpus:
      model_to_states[model_name]['sents'].append(sentence)
      input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True,
                                                 max_length=128)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
      input_ids = input_ids.to(torch.device('cuda'))
      with torch.no_grad():
        output = model(input_ids)
        last_hidden_states = output[0]

        # avg pool last hidden layer
        squeezed = last_hidden_states.squeeze(dim=0)
        masked = squeezed[:input_ids.shape[1], :]
        avg_pooled = masked.mean(dim=0)
        model_to_states[model_name]['states'].append(avg_pooled.cpu())

    end = time.time()
    logger.info('encoded with {} in {} seconds'.format(model_name, end - start))
    np_tensors = [np.array(tensor) for tensor in model_to_states[model_name]['states']]
    model_to_states[model_name]['states'] = np.stack(np_tensors)
  return model_to_states


# encode the training data for each domain
def convert_text_file_to_features(file_path, tokenizer,
                                  max_length=512,
                                  pad_token=0,
                                  pad_token_segment_id=0,
                                  mask_padding_with_zero=True,
                                  max_lines_to_encode=None,
                                  sentencepience=False):
  features = []
  with open(file_path, 'r', encoding='utf-8') as fin:
    for ex_index, line in enumerate(tqdm(fin)):
      if max_lines_to_encode is not None and ex_index >= max_lines_to_encode:
        logger.info("Finish converting {} lines to feature".format(ex_index))
        break
      example_text = line.strip()
      if sentencepience:
        pieces = example_text.split(' ')
        example_text = ''.join(pieces).replace('▁', ' ')

      if ex_index % 10000 == 0:
        logger.info("Writing example %d" % (ex_index))
      inputs = tokenizer.encode_plus(
        text=example_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length)
      input_ids = inputs["input_ids"]
      attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
      padding_length = max_length - len(input_ids)
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
      assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
      features.append(
        InputFeatures(input_ids=input_ids,
                      attention_mask=attention_mask,
                      label=[]))
  return features


def features_to_tensor_dataset(features):
  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
  return dataset


def run_batched_inference(tensor_dataset, batch_size=100):
  # Load pretrained model
  model_class, tokenizer, pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-multilingual-cased')
  model_type = 'distilbert'
  model = model_class.from_pretrained(pretrained_weights)

  # setup device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device == 'cpu':
    logger.info('using cpu!!!')
  else:
    logger.info('using gpu.')
  model.to(device)

  # Start inference loop
  inference_sampler = SequentialSampler(tensor_dataset)
  inference_dataloader = DataLoader(tensor_dataset, sampler=inference_sampler, batch_size=batch_size)
  avg_pooled_all = []
  for batch in tqdm(inference_dataloader, desc="Running inference..."):
    batch = tuple(t.to(device) for t in batch)
    model.eval()
    with torch.no_grad():
      inputs = {'input_ids': batch[0],
                'attention_mask': batch[1]}
      #                       'labels': batch[3]}
      # XLM, DistilBERT and RoBERTa don't use segment_ids
      if model_type != 'distilbert':
        inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None
      outputs = model(**inputs)[0]  # of shape [batch_size, seq_len, state_size]

      # Compute avg pooling
      avg_pooled_batch = []
      for i in range(outputs.shape[0]):
        seq_len = inputs['attention_mask'][i].sum().item()
        avg_pooled_batch.append(outputs[i][range(seq_len), :].mean(dim=0).cpu().numpy())
      avg_pooled_batch = np.stack(avg_pooled_batch)
      avg_pooled_all.append(avg_pooled_batch)
  avg_pooled_all = np.concatenate(avg_pooled_all)
  logger.info(avg_pooled_all.shape)
  return avg_pooled_all


def encode_text_file_and_save(file_path, output_path, max_lines_to_encode, sentencepience=False):
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
  input_features = convert_text_file_to_features(file_path, tokenizer,
                                                 max_length=128,
                                                 max_lines_to_encode=max_lines_to_encode,
                                                 sentencepience=sentencepience)
  tensor_dataset = features_to_tensor_dataset(input_features)
  start = time.time()
  avg_pooled = run_batched_inference(tensor_dataset, batch_size=512)
  end = time.time()
  os.system('mkdir -p {}'.format('/'.join(output_path.split('/')[:-1])))
  logger.info('encoded in {} seconds'.format(end - start))
  np.save(output_path, avg_pooled)
  centroid = np.mean(avg_pooled,axis=0)
  np.save("{}.centroid.npy".format(output_path), centroid)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="input file to compute domain vector")
  parser.add_argument("--output", required=True, help="output file name")
  parser.add_argument("--max-lines", help="max lines to encode")
  parser.add_argument("--sentencepiece", action="store_true", help="Input is in sentence piece, need to detok first")
  args = parser.parse_args()

  input_file = args.input
  output_file = args.output
  max_lines_to_encode = None
  if args.max_lines is not None:
    max_lines_to_encode = args.max_lines

  encode_text_file_and_save(input_file, output_file, max_lines_to_encode, sentencepience=args.sentencepiece)

if __name__ == "__main__":
  main()
