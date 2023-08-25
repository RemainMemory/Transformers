from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

print("please enter the sentence you want to process: ")
input_text = input("")
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Use max_new_tokens instead of max_length
outputs = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

#The sentiment analysis result of Flan-T5 is "positive",
# which more accurately reflects the emotional tendency of the original text.
# The original text mainly discusses the importance of a healthy lifestyle,
# which is overall positive. The sentiment analysis result of the regular T5 model is "True",
# which is not an effective sentiment analysis result.