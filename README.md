# guda
Source code for paper ["Generalised Unsupervised Domain Adaptation of Neural Machine Translation with Cross-Lingual Data Selection"](https://aclanthology.org/2021.emnlp-main.268/) - EMNLP21

### Instalation

```bash
    pip install -r requirements.txt
    cd my_fairseq && pip install --editable ./
```

### Preprare data
Please see `experiments/prepare_data.sh`

### GUDA experiments
#### Train NMT on source task
```shell script
dataset=wmt20_en_de
src=en
tgt=de
./source_task/train_nmt_src.sh $dataset $src $en 
```

#### Data selections
##### Contrastive-based data selection
Prepare training data
````shell script
# subsample source data
./data_selection/contrastive/subsample_general.sh

# binary data - using mBERT wordpiece
./data_selection/binary_data.sh $tgt $dataset

# compute pool avg representation for each sentence
./data_selection/run_compute_vector.sh $tgt $dataset

# clustering the pool avg representation
./data_selection/run_kmeans.sh $tgt $dataset

# train adaptive layer
./data_selection/train_constrastive.sh $tgt $k

# Prepare data to train domain classifier
./data_selection/prepare_domain_disc_data.sh $tgt $dataset

# Train domain classifier
./data_selection/train_domain_disc_contrastive.sh

# Selection
## encode monolingual data 
## Note that the monolingual can be partitioned into multiple sharded
## split -l 1000000 -d mono.$tgt mono.10m.shards.$tgt
./data_selection/encode_mono_wordpiece.sh $tgt
# score
./data_selection/score_en.sh $tgt $domain $k
# merge sharded
python merge_sorted_file.py --input-dir path/to/score --output-file output --file-pattern score.*

# get top 500000 sentences
head -n 500000 output > selected_cons.en

````
##### Cross entropy difference
Train generic language model
```shell script
dataset=lm-encs
suffix=news
./data_selection/ced/train_lm.sh $dataset $suffix
```        

Train in-domain language model
```shell script
dataset=
suffix=
./data_selection/ced/train_lm_indomain.sh $dataset $suffix
```  

Calculate CED and ranking
```shell script
src=en
tgt=de
domain=law
./data_selection/ced/score.sh $tgt
./data_selection/ced/score-and-ced.sh $tgt $domain
```  

#### UDA

```shell script
## Prepare backtranslation data
./data_selection/prepare/backtranslate_all_mono.sh
./data_selection/do_backtranslate.sh $source_dataset_index $domain $gen_subset
## binarize data
./data_selection/prepare/binarize_uda_data.sh $tgt $sel_type

## Run UDA and evaluation
./target_task/run_all_dgx.sh
```

### References
Please cite the following paper if you found the resources in this repository useful.
```
@inproceedings{vu-etal-2021-generalised,
    title = "Generalised Unsupervised Domain Adaptation of Neural Machine Translation with Cross-Lingual Data Selection",
    author = "Vu, Thuy-Trang  and
      He, Xuanli  and
      Phung, Dinh  and
      Haffari, Gholamreza",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.268",
    doi = "10.18653/v1/2021.emnlp-main.268",
    pages = "3335--3346"
}

```
