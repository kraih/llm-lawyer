# LLM-Lawyer Experiment

This is a collection of tools and config files we use to fine-tune, benchmark and deploy Llama-3 as a text classifier
for [Cavil](https://github.com/openSUSE/cavil).

## Background

[Cavil](https://github.com/openSUSE/cavil) uses a pattern matching system to identify potential legal text in source
code. This process is based around identifying hot zones of legal keywords (snippets) and produces around 80% false
positives. Historically these false positives had to be sorted out by humans. So a few years ago we've started to use
a [Character-level Convolutional Network](https://github.com/kraih/Character-level-cnn-pytorch) to automate much of
this process with machine learning. Today we train the model on 150.000 samples, and reach about 96% accuracy. The
model has to be re-trained regularly though to maintain that level.

Large language models such as Llama-3 presented an opportunity for a text classifier that can reach a similar level of
accuracy, but that already has a deep enough understanding of human language that it will require much less
re-training.

## Experiment 1

### Preparation

You need:

1. A checkout of this repo.
2. Copy of the [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) base model.
3. Copy of our 150.000 samples of [training data](https://huggingface.co/datasets/openSUSE/cavil-legal-text).

### Process

```bash
# Install dependencies
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# Convert downloaded Cavil training data back into LegalDB format
./.venv/bin/python revert.py -i legal_text.jsonl -o legaldb-ml-data

# Convert full LegalDB training data to alpaca format (ready for upload to HF)
./.venv/bin/python convert.py -i legaldb-ml-data -o data.json -f alpaca

# Convert subset of LegalDB training data to datasets format (1000 samples of each type for testing)
./.venv/bin/python convert.py -i legaldb-ml-data -o legaldb-ml-data-small.jsonl -f datasets -l 1000

# Test to get a base accuracy (already 75% for Llama-3-8B)
./.venv/bin/python test.py -i legaldb-ml-data-small.jsonl -m /tmp/Meta-Llama-3-8B-Instruct

# Fine-tune Llama-3 with torchtune and LegalDB training data (takes about 8 hours with an RTX 4090)
./.venv/bin/tune run lora_finetune_single_device --config experiment1/torchtune.yaml dataset.source=kraih/legaldb-training-data

# HACK: Convert Llama-3 checkpoint to a format transformers will accept
# (see https://github.com/pytorch/torchtune/issues/832 for more)
cp /tmp/Meta-Llama-8B-Instruct/meta_model_0.pt /tmp/Meta-Llama-3-8B-Instruct/original/consolidated.00.pth
./.venv/bin/python convert_model.py --input_dir /tmp/Meta-Llama-3-8B-Instruct --output_dir /tmp/Meta-Llama-8B-Instruct-Cavil-hf

# Test again to verify improved accuracy (should be about 96% now)
./.venv/bin/python test.py -i legaldb-ml-data-small.jsonl -m /tmp/Meta-Llama-3-8B-Instruct-Cavil-hf

# Start server for use with Cavil
./.venv/bin/python server.py -p 5000 -m /tmp/Meta-Llama-3-8B-Instruct-Cavil-hf

# Verify server works properly, expected result: {"license":true, "confidence":"92.39"}
curl -X POST --data '# MIT License' http://127.0.0.1:5000
```
