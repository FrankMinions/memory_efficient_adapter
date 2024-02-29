import torch
import torch.nn as nn
import torch.utils.checkpoint
from types import MethodType
from transformers.utils import logging
from typing import Optional, Tuple
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import dropout_add, BloomAttention

scaled_dot_product_attention = None

logger = logging.get_logger(__name__)


def _import_sdp_kernel():
    global scaled_dot_product_attention
    try:
        from F import scaled_dot_product_attention as __scaled_dot_product_attention
        scaled_dot_product_attention = __scaled_dot_product_attention
    except ImportError:
        logger.warn(
            f"Your PyTorch version is {torch.__version__}, and torch.nn.functional.Scaled_dot_product_attention, "
            "which supports memory efficient attention and flash attention,"
            "only exists in PyTorch 2.0 and above."
        )


def sdp_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, head_dim, kv_length]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=2)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    _, _, kv_length = key_layer.shape

    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None

    if scaled_dot_product_attention is not None:

        attention_probs = None
        attention_mask = self.beta * alibi

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2)
        output_tensor = self.dense(attn_output)

    else:
        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices): int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices): int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

    output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    outputs = (output_tensor, present)
    if output_attentions:
        outputs += (attention_probs,)

    return outputs


def apply_sdp_kernel(model: nn.Module):
    _import_sdp_kernel()
    for module in model.modules():
        if isinstance(module, BloomAttention):
            module.forward = MethodType(sdp_forward, module)
