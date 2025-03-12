试了一下自己写大语言模型，虽然对rust非常不熟悉（只学了半天），但是还是跌跌撞撞写出来了，感谢AI

整理一下思路
从 safetensors 里获取数据
```
while decoding:
    input = 刚生成的那个token
    for nlayers:
        hidden_states = norm(residual) # hidden_states means temple variables
        q, k, v = hidden_states * wq, wk, wv
        Q = RoPE(x @ Q_weight.T)
        K = RoPE(x @ K_weight.T)
        V = x @ V_weight.T
        K = cat(K_cache, K)
        V = cat(V_cache, V)

        # self_attention:
            score = Q @ K.T / sqrt(dim) # Q_heads = KV_heads * n_groups 
            attn = softmax(score)
            attn_V = attn @ V
            out = attn_V @ O_weight.T
            residual = out + residual

        # Feed-Forward神经网络（mlp函数）
            hidden = rms_norm(residual)
            gate = hidden @ gate_weight.T
            up = hidden @ up_weight.T
            act = gate * sigmoid(gate) * up ## SwiGLU
            output = act @ down_weight.T
            residual = output + residual

    logits = residual * lm_heads # probability of each token
    choose a token
```

还是非常简单的，感叹统计的伟大
    
    
        
        
        
        
