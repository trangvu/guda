#!/usr/bin/env python3

import argparse
import os
from os import path
import zipfile
import urllib.request
import json
import tempfile


from random import randint, shuffle
from xml.parsers.expat import ExpatError

collection_choices = ['ALL']

try:
    with urllib.request.urlopen('http://opus.nlpl.eu/opusapi/?corpora=True') as url:
        data = json.loads(url.read().decode())
        all_collections = data['corpora']
        collection_choices += all_collections
except urllib.error.URLError:
    print('\nWARNING: Could not retrieve corpus list\n')

parser = argparse.ArgumentParser(description='All aboard the OPUS Express! Create test/dev/train sets from OPUS data.')
parser.add_argument('-s', '--src-lang', help='source language (e.g. `en\')', type=str, metavar='lang_id', required=True)
parser.add_argument('-t', '--tgt-lang', help='target language (e.g. `pt\')', type=str, metavar='lang_id', required=True)
parser.add_argument('--test-quota', help='test set size in sentences (default: 10000)', type=int, metavar='num_sents', default=10000)
parser.add_argument('--dev-quota', help='development set size in sentences (default: 10000)', type=int, metavar='num_sents', default=10000)

args = parser.parse_args()

src_lang = args.src_lang
tgt_lang = args.tgt_lang

src_lang, tgt_lang = sorted((src_lang, tgt_lang))

dump_test = []
dump_hiqu = []
dump_loqu = []

test, test_size = [], 0
dev, dev_size = [], 0
train, train_size = [], 0

#####################
## DATA PROCESSING ##
#####################

for collection in collections:
  archive_path = args.root_dir + '/%s/latest/xml/%s-%s.xml.gz' % (collection, src_lang, tgt_lang)

  if not os.path.isfile(archive_path):
    print('Skipping %s (no %s-%s)...' % (collection, src_lang, tgt_lang))

  print('Checking out %s...' % collection)

  try:
    tmp_src = 'tmp_src-%d' % randint(10000000, 99999999)
    tmp_trg = 'tmp_trg-%d' % randint(10000000, 99999999)
    tmp_ids = 'tmp_ids-%d' % randint(10000000, 99999999)

    reader = OpusRead(root_directory=args.root_dir,
            download_dir=args.download_dir,
            attribute='overlap',
            directory=collection,
            source=src_lang, target=tgt_lang,
            write_mode='moses', write=[tmp_src, tmp_trg], write_ids=tmp_ids,
            preserve_inline_tags=args.preserve_inline_tags,
            leave_non_alignments_out=True,
            suppress_prompts=args.q)

    reader.printPairs() # Print data to temp files.

    num_samples = 0

    print('...collating samples...')

    with open(tmp_src, 'r') as srcs:
      with open(tmp_trg, 'r') as trgs:
        with open(tmp_ids, 'r') as ids:
          for src_sent in srcs:
            num_samples += 1
            tgt_sent = trgs.readline()
            id_details = ids.readline().split('\t')

            src_sent = src_sent.strip()
            tgt_sent = tgt_sent.strip()

            src_uri, tgt_uri, src_align, tgt_align, overlap = id_details
            overlap = overlap.strip()

            resource_id = src_uri.split('/')[2] if collection == 'OpenSubtitles' else 0
            one_to_one = len(src_align) == len(tgt_align) == 1
            overlap = float(overlap) if overlap != 'None' else float('-inf')

            if resource_id in test_override:
              dump = dump_test
            elif args.quality_aware and one_to_one and overlap >= args.overlap_threshold:
              dump = dump_hiqu
            else:
              dump = dump_loqu

            doc_id = (src_uri, tgt_uri)
            stats_line = '\t'.join([src_uri, tgt_uri,
                ' '.join(src_align), ' '.join(tgt_align), str(overlap)])

            dump.append((src_sent, tgt_sent, stats_line, doc_id))

    os.remove(tmp_src)
    os.remove(tmp_trg)
    os.remove(tmp_ids)

    if num_samples == 0:
      print('...skipping %s (no %s-%s).' % (collection, src_lang, tgt_lang))

  except ExpatError:
    print('...skipping (ill-formatted XML).')
  except FileNotFoundError:
    print('...skipping (broken links / XML archive not found).')
  except KeyError:
    print('...skipping (broken links / XMLs missing in archive).')
  except Exception as e:
    print(e.args[0])
    print('...skipping (unknown exception).')
  else:
    print('...%d samples processed!' % num_samples)

if not dump_test and not dump_hiqu and not dump_loqu:
  print('Alas, OPUS Express turned up with no data!\n¯\_(ツ)_/¯')
  exit()

if args.shuffle:
  print('Pre-shuffling bins...')

  shuffle(dump_test)
  shuffle(dump_hiqu)
  shuffle(dump_loqu)

  print('...done!')

print('Splitting data into test/dev/train sets...')

for dump in [dump_test, dump_hiqu, dump_loqu]:
  for item in dump:
    if test_size < args.test_quota:
      dump = test
    elif dev_size < args.dev_quota:
      dump = dev
    else:
      dump = train

    if args.doc_bounds:
      if test and item[3] == test[-1][3]:
        dump = test
      elif dev and item[3] == dev[-1][3]:
        dump = dev
      elif train and item[3] == train[-1][3]:
        dump = train

      if dump and item[3] != dump[-1][3]:
        dump.append(('', '', '', None))

    dump.append(item)

    if dump == test:
      test_size += 1 if item != ('', '', '', None) else 0
    elif dump == dev:
      dev_size += 1 if item != ('', '', '', None) else 0
    else:
      train_size += 1 if item != ('', '', '', None) else 0

print('...done!')

if args.shuffle:
  print('Post-shuffling test/dev/train sets...')

  shuffle(test)
  shuffle(dev)
  shuffle(train)

  print('...done!')

test_set = args.test_set

test_src_path = '%s.%s' % (test_set, src_lang)
test_tgt_path = '%s.%s' % (test_set, tgt_lang)
test_ids_path = '%s.%s' % (test_set, 'ids')

if not args.force:
  while path.isfile(test_src_path) or path.isfile(test_tgt_path) or path.isfile(test_ids_path):
    answer = input('Using `--test-set %s\' will cause files to be overwritten. Please input another name, or type OVERWRITE to continue: ' % test_set)

    if answer == 'OVERWRITE':
      break
    else:
      test_set = answer
      test_src_path = '%s.%s' % (test_set, src_lang)
      test_tgt_path = '%s.%s' % (test_set, tgt_lang)
      test_ids_path = '%s.%s' % (test_set, 'ids')

with open(test_src_path, mode='w', encoding='utf-8') as test_src_file:
  with open(test_tgt_path, mode='w', encoding='utf-8') as test_tgt_file:
    with open(test_ids_path, mode='w', encoding='utf-8') as test_ids_file:
      print('Writing test data to `%s.{%s,%s,%s}\'...' % (test_set, src_lang, tgt_lang, 'ids'))

      num_written = 0
      for src_line, tgt_line, ids_line, _ in test:
        test_src_file.write(src_line + '\n')
        test_tgt_file.write(tgt_line + '\n')
        test_ids_file.write(ids_line + '\n')

        num_written += 1
        if num_written % 1000000 == 0:
          print('%d/%d lines written...' % (num_written, len(test)))

      print('...done!')

dev_set = args.dev_set

dev_src_path = '%s.%s' % (dev_set, src_lang)
dev_tgt_path = '%s.%s' % (dev_set, tgt_lang)
dev_ids_path = '%s.%s' % (dev_set, 'ids')

if not args.force:
  while path.isfile(dev_src_path) or path.isfile(dev_tgt_path) or path.isfile(dev_ids_path):
    answer = input('Using `--dev-set %s\' will cause files to be overwritten. Please input another name, or type OVERWRITE to continue: ' % dev_set)

    if answer == 'OVERWRITE':
      break
    else:
      dev_set = answer
      dev_src_path = '%s.%s' % (dev_set, src_lang)
      dev_tgt_path = '%s.%s' % (dev_set, tgt_lang)
      dev_ids_path = '%s.%s' % (dev_set, 'ids')

with open(dev_src_path, mode='w', encoding='utf-8') as dev_src_file:
  with open(dev_tgt_path, mode='w', encoding='utf-8') as dev_tgt_file:
    with open(dev_ids_path, mode='w', encoding='utf-8') as dev_ids_file:
      print('Writing development data to `%s.{%s,%s,%s}\'...' % (dev_set, src_lang, tgt_lang, 'ids'))

      num_written = 0
      for src_line, tgt_line, ids_line, _ in dev:
        dev_src_file.write(src_line + '\n')
        dev_tgt_file.write(tgt_line + '\n')
        dev_ids_file.write(ids_line + '\n')

        num_written += 1
        if num_written % 1000000 == 0:
          print('%d/%d lines written...' % (num_written, len(dev)))

      print('...done!')

train_set = args.train_set

train_src_path = '%s.%s' % (train_set, src_lang)
train_tgt_path = '%s.%s' % (train_set, tgt_lang)
train_ids_path = '%s.%s' % (train_set, 'ids')

if not args.force:
  while path.isfile(train_src_path) or path.isfile(train_tgt_path) or path.isfile(train_ids_path):
    answer = input('Using `--train-set %s\' will cause files to be overwritten. Please input another name, or type OVERWRITE to continue: ' % train_set)

    if answer == 'OVERWRITE':
      break
    else:
      train_set = answer
      train_src_path = '%s.%s' % (train_set, src_lang)
      train_tgt_path = '%s.%s' % (train_set, tgt_lang)
      train_ids_path = '%s.%s' % (train_set, 'ids')

with open(train_src_path, mode='w', encoding='utf-8') as train_src_file:
  with open(train_tgt_path, mode='w', encoding='utf-8') as train_tgt_file:
    with open(train_ids_path, mode='w', encoding='utf-8') as train_ids_file:
      print('Writing training data to `%s.{%s,%s,%s}\'...' % (train_set, src_lang, tgt_lang, 'ids'))

      num_written = 0
      for src_line, tgt_line, ids_line, _ in train:
        train_src_file.write(src_line + '\n')
        train_tgt_file.write(tgt_line + '\n')
        train_ids_file.write(ids_line + '\n')

        num_written += 1
        if num_written % 1000000 == 0:
          print('%d/%d lines written...' % (num_written, len(train)))

      print('...done!')

