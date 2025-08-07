## Environment Setup
```
conda create --name qwenvl python=3.11
pip install -r qwen-vl-finetune/requirements.txt
pip install flash_attn
```

## Training

Lora training
```
qwen-vl-finetune/scripts/lora_7b.sh
```
```
qwen-vl-finetune/scripts/lora_72b.sh
```

Full-finetuning
```
qwen-vl-finetune/scripts/sft_32b.sh
```
```
qwen-vl-finetune/scripts/sft_72b.sh
```
