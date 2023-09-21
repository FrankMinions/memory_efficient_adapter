# memory_efficient_adapter

Since the model structure scripts provided by transformers are not adapted to flash attention and memory efficient attention, users need to modify the source code themselves if they want to use the above two solutions and embed them in the process of calculating multi-head attention. This undoubtedly increases the user's workload and raises the threshold for using GPU memory optimization.

Therefore, I have done this work for the user in advance and embedded xformers and scaled_dot_product_attention in the corresponding positions of the code. 

Since the framework provided by Dao-AILab only supports Ampere, Ada, or Hopper GPUs, and it does not support specifying attention bias, I chose the above in terms of versatility.

However, you need to check your PyTorch version and make sure you have xformers installed.

Currently, the project has adapted BLOOM, BART, LLaMA and Qwen LLM.

Stay tuned for other models!

How to use this GPU memory adapter?

- step1: 
```bash
pip install -U xformers

pip install torch torchvision torchaudio
```
Please check again whether the installed PyTorch version is 2.0 or above.

- step2:

Please declare a global variable in your training, fine-tuning or inference shell scripts:

```bash
export  USE_FLASH_ATTN=true

torchrun --nproc_per_node=1 --nnodes=8 xxx.py \
  --use_flash_attn USE_FLASH_ATTN \
  ...
```
Before doing above you need to embed the following in your python script:
```python
if training_args.use_flash_attn:
    
    # if you want to train llama
    from memory_efficient_adapter.models.llama.flash_attn_patch import apply_attention_patch
    apply_attention_patch()
```
The same applies to other models.
