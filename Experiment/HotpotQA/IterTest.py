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

Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern 
Records and born in December 5, 1932?
Answer: Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and 
John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, 
singer, actor, comedian, and songwriter.
So the final answer is: Little Richard

Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
Answer: Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine 
biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had 
more diverse jobs than Rachel Carson.
So the final answer is: Chinua Achebe

Question: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?
Answer: Remember Me Ballin' is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell's stage name, 
who was born in August 7, 1979, and is an American rapper.
So the final answer is: 1979

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

with open("HotpotQA_sample_100.json", "r", encoding="utf-8") as f:
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

output_file = "hotpot_Iter_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)