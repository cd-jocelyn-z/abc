

## Get Started

### Set up the folder and dowload the data
```
bash setup.sh
```

Note: the data for the ALL-mixture will be downloaded, the filtered dataset Ofp-Arbre is made available in the drive

###  Set up the enviornement
```
pip install requirements.txt
```

### Tune

Example Usage:
```
python nllb-bre-fra-lora.py --train data/train.jsonl --dev data/dev.jsonl --test data/test.jsonl --output-dir "encoder_crossattn_full_3ep" --config module_configs.json --lora-config "encoder_crossattn" --epoch 3
```


https://drive.google.com/drive/folders/1HvCYGiZOhZy7FPPsp8qgJK3W8rCnciBO?usp=sharing
