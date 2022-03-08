#!/bin/bash

##### Prepare source domain data - WMT
./preprocess/prepare-wmt14en2fr.sh
./preprocess/prepare-wmt20en2cs.sh
./preprocess/prepare-wmt20en2de.sh
./preprocess/prepare-wmt19en2fi.sh
./preprocess/prepare-ar-en.sh

##### Prepare target domain data
./preprocess/prepare-tgt-en-ar.sh
./preprocess/prepare-tgt-en-cs.sh
./preprocess/prepare-tgt-en-de.sh
./preprocess/prepare-tgt-en-fr.sh

##### Prepare monolingual data
./preprocess/prepare-mono-en.sh

##### Apply sentencepiece and binarize data
./preprocess/fs_preprocess.sh
./preprocess/fs_preprocess_tgt_domain.sh

# For Ar-En, src and tgt languages don't share the vocabulary
./preprocess/fs_preprocess_tgt_domain_arabic.sh