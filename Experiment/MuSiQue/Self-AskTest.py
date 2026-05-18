import time
import re
import string
import json
from openai import OpenAI

deepseek_api_key = "sk-bb011033f3fa41acbdae135831ae0e6d"
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

selfask_prompt_template = """ 
Question: Who lived longer, Muhammad Ali or Alan Turing? 
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born? 
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: United States.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
So the final answer is: No

Question: {question}
Are follow up questions needed here:

IMPORTANT:
You MUST strictly follow this exact format:
You must always end your reasoning with:
So the final answer is: <answer>
"""

def extract_answer(text):
    text = text.replace('*', '').replace('_', '')

    match = re.search(
        r"So the final answer is:\s*(.*?)(?:\.|\n|$)",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    lines = text.strip().split("\n")
    if lines:
        return lines[-1].strip()

    return text.strip()

def normalize(text):
    text = str(text).lower().strip()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text)

    return text

def compute_em(pred, gold1, gold2=None):
    pred = normalize(pred)
    gold1 = normalize(gold1)

    if pred == gold1:
        return 1

    if gold2 and normalize(gold2) == pred:
        return 1

    return 0

with open("musique_sample_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
ems = []

for idx, row in enumerate(data):

    question = row["question"]
    gold = row["answer"]

    print(f"\n==== {idx} ====")
    print("Question:", question)

    prompt = selfask_prompt_template.format(question=question)

    try:
        generated = ask_llm(prompt)
        print("Generated:\n", generated)

        pred = extract_answer(generated)

    except Exception as e:
        print("Error:", e)
        pred = ""

    print("Predicted Answer:", pred)

    em = compute_em(pred, gold)
    print("EM:", em)

    result = {
        "id": row["id"],
        "question": question,
        "gold_answer": gold,
        "prediction": pred,
        "em": em
    }

    results.append(result)
    ems.append(em)

    time.sleep(1)

output_file = "musique_selfask_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)