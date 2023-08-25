# Import the required libraries and modules
# We import the necessary libraries and modules to work with the T5 model,
# tokenizer, and Spacy for entity extraction.

from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy

# Initialize T5 model and tokenizer
# We initialize the T5 model and tokenizer.
# T5 is a powerful pre-trained model used for text generation tasks.

tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Load Spacy's English model
# We load Spacy's English model, which will be used for entity extraction.

nlp = spacy.load('en_core_web_sm')

# Function for generating summaries
# This function is responsible for generating summaries of input text.
def summarize(text):
    # Encode the input text with the T5 tokenizer and add the 'summarize:' prefix.
    # The tokenizer converts the text into numerical tokens (input IDs).
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate a summary using the T5 model.
    # We set the maximum length of the generated summary to 150 tokens and the minimum length to 40 tokens.
    # The length penalty of 2.0 encourages longer summaries, and we use 4 beams for search to improve diversity.
    # Early stopping allows the generation to stop once the specified conditions are met.
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary and return it as a human-readable text.
    return tokenizer.decode(outputs[0])

# Function for sentiment analysis
# This function performs sentiment analysis on the input text.
def sentiment(text):
    # Encode the input text with the T5 tokenizer and add the 'sentiment:' prefix.
    inputs = tokenizer.encode("sentiment: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate a sentiment result using the T5 model.
    # We set the maximum length of the generated result to 2 tokens (positive or negative sentiment).
    # We use 4 beams for search and enable early stopping.
    outputs = model.generate(inputs, max_length=2, num_beams=4, early_stopping=True)

    # Decode the generated sentiment result and return it as a human-readable text.
    return tokenizer.decode(outputs[0])

# Function for question generation
# This function generates a question related to the input text.
def question(text):
    # Encode the input text with the T5 tokenizer and add the 'question:' prefix.
    inputs = tokenizer.encode("question: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate a question using the T5 model.
    # We set the maximum length of the generated question to 150 tokens and the minimum length to 40 tokens.
    # The length penalty of 2.0 encourages longer questions, and we use 4 beams for search to improve diversity.
    # Early stopping allows the generation to stop once the specified conditions are met.
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated question and return it as a human-readable text.
    return tokenizer.decode(outputs[0])

# Function for entity extraction
# This function performs entity extraction on the input text using Spacy.
def extract_entities(text):
    # Use Spacy's English model to extract entities from the input text.
    doc = nlp(text)

    # Print all the extracted entities and their labels (e.g., person, organization, etc.).
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


# Function for translation
# This function translates the input text from English to German.
def translate(text):
    # Encode the input text with the T5 tokenizer and add the 'translate English to German:' prefix.
    inputs = tokenizer.encode("translate English to German: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the translation using the T5 model.
    # We set the maximum length of the generated translation to 512 tokens.
    # We use 4 beams for search and enable early stopping.
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)

    # Decode the generated translation and return it as a human-readable text, skipping special tokens.
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


# Get user input
# Ask the user to input the text they want to process.
text = input("Please enter the text you want to process: ")

# Execute functions and print results
# Call the functions one by one and print the results.
print("Summary:", summarize(text))
print("Sentiment:", sentiment(text))
print("Question:", question(text))
print("Entities:")
extract_entities(text)
print("Translation:", translate(text))
