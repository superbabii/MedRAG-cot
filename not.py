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
    device_map="auto",  # Automatically map model to available devices
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

    # Prepare the prompt by combining the question and options
    prompt = f"Question: {question}\n"
    for option_key, option_value in options.items():
        prompt += f"{option_key}: {option_value}\n"
    
    prompt += "Select the correct answer (A, B, C, or D):\n"

    # Generate text with the model using the prompt
    result = pipe(
        prompt, 
        max_new_tokens=100,  # Replace max_length with max_new_tokens
        num_return_sequences=1,
        do_sample=True,  
        top_k=50,  
        top_p=0.95,  
        temperature=0.7,
        truncation=True,
    )

    # Extract and print the generated answer
    generated_text = result[0]['generated_text']
    # Extract the actual answer (first character after "Answer:")
    generated_answer = ""
    for line in generated_text.split("\n"):
        if line.startswith("Answer:"):
            generated_answer = line.split(":")[1].strip()[0]
            break
    
    is_correct = correct_answer == generated_text[0]
    if is_correct:
        correct_count += 1

    result = {
        'question': question,
        'correct_answer': correct_answer,
        'generated_answer': generated_text,
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
