use safetensors::SafeTensors;

#[test]
pub fn test_names() {
  let model_file = std::fs::read("/home/lyx2/Desktop/llm-rs/models/story/model.safetensors").unwrap();
  let safetensor = SafeTensors::deserialize(&model_file).unwrap();
  println!("{:?}", SafeTensors::read_metadata(&model_file).unwrap());
  println!("{:?}", safetensor.names());
}