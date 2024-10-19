# XTTSv2 Finetuning Guide for New Languages

This guide provides instructions for finetuning XTTSv2 on a new language, using Vietnamese (`vi`) as an example.

[UPDATE] A finetuned model for Vietnamese is now available at [anhnh2002/vnTTS](https://huggingface.co/anhnh2002/vnTTS) on Hugging Face


## Table of Contents
1. [Installation](#1-installation)
2. [Data Preparation](#2-data-preparation)
3. [Pretrained Model Download](#3-pretrained-model-download)
4. [Vocabulary Extension and Configuration Adjustment](#4-vocabulary-extension-and-configuration-adjustment)
5. [DVAE Finetuning (Optional)](#5-dvae-finetuning-optional)
6. [GPT Finetuning](#6-gpt-finetuning)
7. [Usage Example](#7-usage-example)

## 1. Installation

First, clone the repository and install the necessary dependencies:

```
git clone https://github.com/nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages.git
cd XTTSv2-Finetuning-for-New-Languages
pip install -r requirements.txt
```

## 2. Data Preparation

Ensure your data is organized as follows:

```
project_root/
├── datasets-1/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
├── datasets-2/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
...
│   
├── recipes/
├── scripts/
├── TTS/
└── README.md
```

Format your `metadata_train.csv` and `metadata_eval.csv` files as follows:

```
audio_file|text|speaker_name
wavs/xxx.wav|How do you do?|@X
wavs/yyy.wav|Nice to meet you.|@Y
wavs/zzz.wav|Good to see you.|@Z
```

## 3. Pretrained Model Download

Execute the following command to download the pretrained model:

```bash
python download_checkpoint.py --output_path checkpoints/
```

## 4. Vocabulary Extension and Configuration Adjustment

Extend the vocabulary and adjust the configuration with:

```bash
python extend_vocab_config.py --output_path=checkpoints/ --metadata_path datasets/metadata_train.csv --language vi --extended_vocab_size 2000
```

## 5. DVAE Finetuning (Optional)

To finetune the DVAE, run:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=512 \
--lr=5e-6
```

## 6. GPT Finetuning

For GPT finetuning, execute:

[OUTDATED]
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=8 \
--grad_acumm=2 \
--max_text_length=250 \
--max_audio_length=255995 \
--weight_decay=1e-2 \
--lr=5e-6 \
--save_step=2000
```
[UPDATE - Supports training multiple datasets. Format metadatas parameter as follows: `path_to_train_csv_dataset-1,path_to_eval_csv_dataset-1,language_dataset-1 path_to_train_csv_dataset-2,path_to_eval_csv_dataset-2,language_dataset-2 ...`]
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
--output_path checkpoints/ \
--metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,vi datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,vi \
--num_epochs 5 \
--batch_size 8 \
--grad_acumm 4 \
--max_text_length 400 \
--max_audio_length 330750 \
--weight_decay 1e-2 \
--lr 5e-6 \
--save_step 50000
```

## 7. Usage Example

Here's a sample code snippet demonstrating how to use the finetuned model:

```python
import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths
xtts_checkpoint = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
xtts_config = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
tts_text = "Good to see you."
speaker_audio_file = "ref.wav"
lang = "vi"

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Play audio (for Jupyter Notebook)
from IPython.display import Audio
Audio(out_wav, rate=24000)
```

Note: Finetuning the HiFiGAN decoder was attempted but resulted in worse performance. DVAE and GPT finetuning are sufficient for optimal results.

Update: If you have enough short texts in your datasets (about 20 hours), you do not need to finetune DVAE.
