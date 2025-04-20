# DL-Assignment-2

## Overview

This repository contains solutions to two deep learning tasks, focusing on:

1. **Seq2Seq model**: Implementing and training a seq2seq model  on the daskhina dataset.
2. **Fine tuning GPT2**: Fine tuning GPT2 model to generate lyrics of a song given a small part of it 

## Repository Structure

- `DL_assignment_2_Question_1.ipynb`: Notebook for training a Seq2Seq model.
- `DL_assignment_2_Question_2.ipynb`: Notebook for Fine tuning GPT2.
- `DL Assignment 2 Question 2.py`: Fall back Python file for GPT2 model if ipynb is not visible.
- `README.md`: Project documentation.

## Requirements

Ensure you have the following dependencies installed for the Seq2Seq model

```bash
pip install tensorflow numpy pandas tarfile 
```

## Running the Models

### 1. Training the Seq2Seq model 

To train  model:

1. Open `DL_assignment_2_Question_1.ipynb` in Jupyter Notebook.
2. Run the notebook cells sequentially to:
   - Load and preprocess the dakshina dataset.
   - Define the seq2seq model architecture.
   - Compile and train the model.
   - U can find some three models to run and test on
3.Models Used:
  - Model A : A single Layer LSTM model having 188,226 parameters and having an accuracy of 78% on validation data
  - Model B : A single Layer RNN based model having  53,442 parameters and having an accuracy of 74% on the validation data
  - Model C : A Deep LSTM based model having 451,394 parameters and having an accuracy of 84% on the validation data

## 2. GPT2 Fine Tuning

### Requirements
Install all necessary libraries:

```bash
pip install transformers datasets tqdm
pip install torch torchvision torchaudio
pip install kagglehub
```

### Dataset
The script uses the Poetry Dataset by Paul Timothy Mooney from Kaggle, downloaded via kagglehub.

```python
import kagglehub
download_path = kagglehub.dataset_download("paultimothymooney/poetry")
```
All poem/lyric files are read into a single list and used to create a Hugging Face Dataset.

### Model and Tokenization
  -Model: gpt2 from Hugging Face's model hub.
  -Tokenizer: GPT2Tokenizer with the pad_token set as the eos_token (since GPT-2 doesn't include a padding token by default).

The model is fine-tuned using masked language modeling with tokenized lyric text.

### Running the Script
Ensure you have the required libraries installed.
Run the script in A cloud based platform with GPU enabled 

### Example output
``` python
Generated Lyrics

Eighteen years eighteen years
She got one of your kids, got you for eighteen years
In the eighteenth hour, I felt the rhythm start
A melody inside my heart
```
