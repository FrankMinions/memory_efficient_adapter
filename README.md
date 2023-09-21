# memory_efficient_adapter

Since the model structure scripts provided by transformers are not adapted to flash attention and memory efficient attention, users need to modify the source code themselves if they want to use the above two solutions and embed them in the process of calculating multi-head attention. This undoubtedly increases the user's workload and raises the threshold for using GPU memory optimization.

Therefore, I have done this work for the user in advance and embedded xformers and scaled_dot_product_attention in the corresponding positions of the code. 

However, you need to check your PyTorch version and make sure you have xformers installed.

Currently, the project has adapted BLOOM, BART, LLaMA and Qwen LLM.

Stay tuned for other models!
