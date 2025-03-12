use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

/* LEARN 在 Rust 中，impl 是一个关键字，用于定义一个类型（如结构体、枚举或 trait）
的具体实现（implementation）。通过 impl 块，你可以为类型添加方法、关联函数、关联常量等。
你这些方法可以是实例方法（需要一个 self 参数）或关联函数（类似于静态方法）。 */

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        /* LEARN 在 Rust 中，impl Fn 是一种匿名函数类型，
        通常用于定义闭包（closure）或其他函数类型的变量。你的代码片段： */
        let get_tensor: Box<dyn Fn(&str) -> Tensor<f32>> = Box::new(|name: &str| {
            // copy from https://github.com/huggingface/candle/blob/main/candle-core/src/safetensors.rs
            let now_safetensor = safetensor.tensor(name).expect(&format!("cannot get tensor with name {}", name));
            let data = now_safetensor.data();
            let shape = now_safetensor.shape();
            let size_in_bytes = 4;
            let elem_count = data.len() / size_in_bytes;
            if (data.as_ptr() as usize) % size_in_bytes == 0 {
                let data_trans: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count) };
                Tensor::<f32>::new(data_trans.to_vec(), &shape.to_vec())
            } else { // 在非对齐情况下，通过 std::ptr::copy_nonoverlapping 手动复制数据，避免直接操作未对齐的内存。
                let mut c: Vec<f32> = Vec::with_capacity(elem_count);
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
                    c.set_len(elem_count)
                }
                Tensor::<f32>::new(c, &shape.to_vec())
            }
        });

        let get_tensor_vertor = |name: &str, num: usize| -> Vec<Tensor<f32>> {
            (0..num).map(|i| get_tensor(&format!("model.layers.{i}.{name}"))).collect()
        };
        
        LLamaParams {
        /* total names: ["model.layers.0.self_attn.v_proj.weight", "model.layers.1.self_attn.q_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.self_attn.q_proj.weight", "model.norm.weight", "model.layers.1.post_attention_layernorm.weight", "lm_head.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.post_attention_layernorm.weight", "model.layers.1.self_attn.o_proj.weight", "model.layers.1.mlp.up_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.1.input_layernorm.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight", "model.layers.1.self_attn.v_proj.weight", "model.layers.1.self_attn.k_proj.weight", "model.layers.1.mlp.gate_proj.weight"] */
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_tensor_vertor("input_layernorm.weight", config.num_hidden_layers),
            wq: get_tensor_vertor("self_attn.q_proj.weight", config.num_hidden_layers),
            wk: get_tensor_vertor("self_attn.k_proj.weight", config.num_hidden_layers),
            wv: get_tensor_vertor("self_attn.v_proj.weight", config.num_hidden_layers),
            wo: get_tensor_vertor("self_attn.o_proj.weight", config.num_hidden_layers),
            rms_ffn_w: get_tensor_vertor("post_attention_layernorm.weight", config.num_hidden_layers),
            w_up: get_tensor_vertor("mlp.up_proj.weight", config.num_hidden_layers),
            w_gate: get_tensor_vertor("mlp.gate_proj.weight", config.num_hidden_layers),
            w_down: get_tensor_vertor("mlp.down_proj.weight", config.num_hidden_layers),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
