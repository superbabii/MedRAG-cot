import torch
from transformers import pipeline
import json
import random

# Load the benchmark JSON file
with open('benchmark.json', 'r') as f:
    benchmark_data = json.load(f)

# Get 5 random questions
random_questions = random.sample(list(benchmark_data.items()), 1)

# Ensure the appropriate device is selected (GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Load the language model
pipe = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-1B",  
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  
)
print(f"Model loaded successfully.")

# Store the results of comparisons
results = []
correct_count = 0

# Iterate over each question and get the generated answer
for question_id, question_data in random_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    # Prepare the refined prompt by combining the question and options in a structured manner
    prompt = f"""You are given a multiple-choice question. Please provide the correct answer by choosing one of the options (A, B, C, or D).
    
Question:
{question}

Options:
"""
    for option_key, option_value in options.items():
        prompt += f"{option_key}: {option_value}\n"
    
    prompt += """\nPlease provide your answer as a single letter (A, B, C, or D)."""

    # Generate text with the model using the refined prompt
    result = pipe(
        prompt, 
        max_new_tokens=10,  # Limit to short responses
        num_return_sequences=1,  
        do_sample=True,  
        top_k=50,  
        top_p=0.95,  
        temperature=0.7,
        truncation=True,
    )

    # Extract and print the generated answer
    generated_text = result[0]['generated_text']
    
    print(generated_text)
    print('-' * 50)

    # Extract the first character of the generated text, assuming it's the answer
    generated_answer = generated_text.strip().split()[-1][0]  # Get the first character
    
    is_correct = correct_answer == generated_answer
    if is_correct:
        correct_count += 1

    result = {
        'question': question,
        'correct_answer': correct_answer,
        'generated_answer': generated_answer,
        'is_correct': is_correct
    }
    results.append(result)
    
# Print the results of the comparison
for result in results:
    print(f"Question: {result['question']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print(f"Is Correct: {result['is_correct']}")
    print('-' * 50)

# Calculate accuracy
accuracy = correct_count / len(results) * 100
print(f"Accuracy: {accuracy:.2f}%")
