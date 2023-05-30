This is the official codes for Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model.
**FOR REVIEW ONLY, DO NOT DISTRIBUTE**

# Dependency

```angular2html
datasets==1.6.2 
scikit-learn==0.24.2 
tensorboard==2.5.0 
matplotlib==3.4.2 
transformers==4.6.0 
numpy==1.21.1 
```

# Setup 
```bash
pip sinstall -v -e .
```

# Run GLUE Experiments

Run DIVISION on T5 language models:

```bash 
for dataset in ("rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")
do
    for seed in (0 1 2)
    do
    bash scripts/approx_linear.sh 0 $dataset 1 1 1 1 1 1 8 4 division
    done
done
```

## Acknowledgment
Our code is based on the official code of [Ladder Site Tuning](https://arxiv.org/abs/2206.06522)