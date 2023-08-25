from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

print("please enter the sentence you want to process: ")
input_text = input("")
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Use max_new_tokens instead of max_length
outputs = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
