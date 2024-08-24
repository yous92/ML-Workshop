from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

question = "What is the capital of France?"

inputs = tokenizer.encode(question, return_tensors="pt")

outputs = model.generate(inputs, max_length=18, num_return_sequences=1)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


