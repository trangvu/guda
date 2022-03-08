#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import argparse
import logging
import os
import time
from operator import itemgetter

import torch
from torch.serialization import default_restore_location

from fairseq import options, tasks, utils
from fairseq.data import LMContextWindowDataset, data_utils
from fairseq.file_io import PathManager
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, InputFeatures

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_pretrained_model(model, pretrained_path):
    bexists = PathManager.isfile(pretrained_path)
    if bexists:
        with PathManager.open(pretrained_path, "rb") as f:
            state = torch.load(
                f, map_location=lambda s, l: default_restore_location(s, "cpu")
            )
        logger.info("Load model from {}".format(pretrained_path))
        model.load_state_dict(
            state["model"], strict=True
        )
    else:
        logger.warning("Cannot file checkpoint {}".format(pretrained_path))
        return model
    return model

def features_to_tensor_dataset(features):
  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
  return dataset


def run_batched_inference(args, model, tensor_dataset, device, classification_head_name, batch_size=100):
  # Start inference loop
  inference_sampler = SequentialSampler(tensor_dataset)
  inference_dataloader = DataLoader(tensor_dataset, sampler=inference_sampler, batch_size=batch_size)
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
  idx = -1
  score_output = []

  for batch in tqdm(inference_dataloader, desc="Running inference..."):
    batch = tuple(t.to(device) for t in batch)
    model.eval()
    seq_len = batch[1].sum(axis=1)
    with torch.no_grad():
      inputs = {'src_tokens': batch[0],
                'src_lengths': seq_len,
                'classification_head_name': classification_head_name,
                'adaptive_layer_name': 'adaptive',
                'source_only': True}
      outputs, _ = model(**inputs) # of shape [batch_size, seq_len, state_size]
      log_prob = outputs.softmax(1).detach().cpu().numpy()
      predictions = log_prob.argmax(1)
      score = log_prob.max(1)

      for i in range(len(score)):
        idx += 1
        if predictions[i] == 1:
          tok = tokenizer.convert_ids_to_tokens(batch[0][i][:seq_len[i]])
          detok = ' '.join(tok).replace(' ##', '')
          score_output.append((idx, score[i], detok))

  score_output.sort(key=itemgetter(1), reverse=True)

  with open(args.score_output, 'w') as fout:
    for (id, score, sent) in score_output:
      fout.write("{}\t{:2f}\t{}\n".format(idx, score, sent))

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
          truncation=True,
        add_special_tokens=False,
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

def main(args, **unused_kwargs):
    bexists = PathManager.isfile(args.model_path)
    state_dict = {}
    if bexists:
      with PathManager.open(args.model_path, "rb") as f:
        state_dict = torch.load(
          f, map_location=lambda s, l: default_restore_location(s, "cpu")
        )
      logger.info("Load model from {}".format(args.model_path))
    else:
      raise("Cannot file checkpoint {}".format(args.model_path))

    model_args = state_dict['args']

    device = 'cpu'
    if torch.cuda.is_available() and not model_args.cpu:
        torch.cuda.set_device(model_args.device_id)
        device = "cuda:{}".format(model_args.device_id)

    utils.import_user_module(args)
    model_args.data = args.data
    logger.info(model_args)

    use_cuda = torch.cuda.is_available() and not model_args.cpu

    task = tasks.setup_task(model_args)

    # Load ensemble
    model = task.build_model(model_args)
    model.load_state_dict(
            state_dict["model"], strict=True
        )

    if model_args.fp16:
        model.half()
    if use_cuda:
        model.cuda()


    logger.info('num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    input_features = convert_text_file_to_features(args.score_input, tokenizer,
                                                   max_length=128,
                                                   sentencepience=args.sentencepiece)
    tensor_dataset = features_to_tensor_dataset(input_features)
    start = time.time()
    run_batched_inference(args, model, tensor_dataset, device, model_args.classification_head_name, batch_size=args.batch_size)
    end = time.time()
    logger.info('encoded in {} seconds'.format(end - start))

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help="Model path")
    parser.add_argument('--score-output', required=True, help='Output score file')
    parser.add_argument('--score-input', required=True, help='Raw file to score')
    parser.add_argument("--sentencepiece", action="store_true", help="Input is in sentence piece, need to detok first")
    parser.add_argument("--user-dir", help="User dir")
    parser.add_argument("--data", help="data-bin")
    parser.add_argument("--batch-size", type=int, default=10, help="data-bin")
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
