import time
import re
import string
import json
from openai import OpenAI

deepseek_api_key = ""
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

cot_prompt_template = """

Question: Who lived longer, Muhammad Ali or Alan Turing?
Answer: Muhammad Ali was 74 years old when he died. Alan Turing was 41 years old when he died. Therefore, Muhammad Ali lived longer.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Answer: The founder of Craigslist was Craig Newmark. Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Are both the directors of Jaws and Casino Royale from the same country?
Answer: The director of Jaws is Steven Spielberg, who is from the United States. The director of Casino Royale is Martin Campbell, who is from New Zealand. 
The United States and New Zealand are not the same country.
So the final answer is: No

Question: {question}
Answer:
So the final answer is:

IMPORTANT:
You MUST strictly follow this exact format:
You must always end your reasoning with:
So the final answer is: <answer>
"""

ITER_N_PROMPT = """You are given a question and a previous reasoning attempt.

Your task:
1. Carefully review the previous reasoning
2. Identify any errors or missing information
3. Improve the reasoning step by step
4. Provide a more accurate final answer

IMPORTANT:
- Do NOT blindly trust the previous reasoning
- If it is wrong, correct it
- If information is missing, infer it carefully

Previous reasoning attempt:
{prev_generation}

Question: {question}

Answer: Let's think step by step.
So the final answer is:

IMPORTANT: Always end with exactly: So the final answer is: <answer>
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

with open("2wiki_sample_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
ems = []

for idx, row in enumerate(data):

    question = row["question"]
    gold = row["answer"]

    print(f"\n==== {idx} ====")
    print("Question:", question)

    try:
        prompt = cot_prompt_template.format(question=question)
        y_prev = ask_llm(prompt)

        print("\n[Iteration 1]")
        print(y_prev)

        T = 2

        for t in range(2, T + 1):

            iter_prompt = ITER_N_PROMPT.format(
                prev_generation=y_prev,
                question=question
            )

            y_new = ask_llm(iter_prompt)

            print(f"\n[Iteration {t}]")
            print(y_new)

            if extract_answer(y_new) == extract_answer(y_prev):
                print("Early stop (no change in answer)")
                break

            y_prev = y_new

        generated = y_prev
        pred = extract_answer(generated)

    except Exception as e:
        print("Error:", e)
        pred = ""

    print("Predicted Answer:", pred)

    em = compute_em(pred, gold)
    print("EM:", em)

    result = {
        "id": row["_id"],
        "question": question,
        "gold_answer": gold,
        "prediction": pred,
        "em": em
    }

    results.append(result)
    ems.append(em)

    time.sleep(1)

output_file = "2wiki_Iter_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)