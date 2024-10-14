import json
import random
from src.medrag import MedRAG

# Load the benchmark JSON file
with open('benchmark.json', 'r') as f:
    benchmark_data = json.load(f)

# Get 5 random questions
random_questions = random.sample(list(benchmark_data.items()), 5)

# Initialize the MedRAG system
medrag_system = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=False)

# Store the results of comparisons
results = []
correct_count = 0

# Iterate over each question and get the generated answer
for question_id, question_data in random_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    # Use MedRAG to generate the answer
    generated_answer, _, _ = medrag_system.answer(question=question, options=options)

    # Parse the generated answer if it's a dictionary
    try:
        generated_answer_dict = json.loads(generated_answer)  # Convert the string to a dictionary
        generated_choice = generated_answer_dict['answer_choice'].strip('" ')  # Extract and clean the answer choice
    except (json.JSONDecodeError, KeyError):
        generated_choice = generated_answer.strip('" ')  # Fallback to the original format if not a dictionary

    # Compare the generated answer with the correct one
    is_correct = correct_answer == generated_choice
    if is_correct:
        correct_count += 1

    result = {
        'question_id': question_id,
        'question': question,
        'correct_answer': correct_answer,
        'generated_answer': generated_choice,
        'is_correct': is_correct
    }
    results.append(result)

# Print the results of the comparison
for result in results:
    print(f"Question ID: {result['question_id']}")
    print(f"Question: {result['question']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print(f"Is Correct: {result['is_correct']}")
    print('-' * 50)

# Calculate accuracy
accuracy = correct_count / len(results) * 100
print(f"Accuracy: {accuracy:.2f}%")
