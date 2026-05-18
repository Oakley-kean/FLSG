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

DAG_PROMPT = """
Analyze the following multi-hop question and decompose it into a reasoning graph.
Rules:
1. Each node is a simple single-hop sub-question
2. A node depends on another if it needs that node's answer to form its question
3. Nodes with no dependencies can be answered independently

Question: Who lived longer, Muhammad Ali or Alan Turing?
Graph decomposition needed: Yes.

NODES:
- N1: How old was Muhammad Ali when he died?
- N2: How old was Alan Turing when he died?
- N3: Who lived longer, [N1_answer] years or [N2_answer] years?

EDGES:
- N3 depends on N1
- N3 depends on N2

EXECUTION_ORDER:
- Round 1 (parallel): N1, N2
- Round 2: N3

EXECUTION:
Round 1:
  N1: How old was Muhammad Ali when he died?
  Answer: Muhammad Ali was 74 years old when he died.

  N2: How old was Alan Turing when he died?
  Answer: Alan Turing was 41 years old when he died.

Round 2:
  N3: Who lived longer, 74 years or 41 years?
  Answer: Muhammad Ali lived longer.

So the final answer is: Muhammad Ali

---

Question: When was the founder of craigslist born?
Graph decomposition needed: Yes.

NODES:
- N1: Who was the founder of craigslist?
- N2: When was [N1_answer] born?

EDGES:
- N2 depends on N1

EXECUTION_ORDER:
- Round 1: N1
- Round 2: N2

EXECUTION:
Round 1:
  N1: Who was the founder of craigslist?
  Answer: Craigslist was founded by Craig Newmark.

Round 2:
  N2: When was Craig Newmark born?
  Answer: Craig Newmark was born on December 6, 1952.

So the final answer is: December 6, 1952

---

Question: Are both the directors of Jaws and Casino Royale from the same country?
Graph decomposition needed: Yes.

NODES:
- N1: Who is the director of Jaws?
- N2: Who is the director of Casino Royale?
- N3: Where is [N1_answer] from?
- N4: Where is [N2_answer] from?
- N5: Are [N3_answer] and [N4_answer] the same country?

EDGES:
- N3 depends on N1
- N4 depends on N2
- N5 depends on N3
- N5 depends on N4

EXECUTION_ORDER:
- Round 1 (parallel): N1, N2
- Round 2 (parallel): N3, N4
- Round 3: N5

EXECUTION:
Round 1:
  N1: Who is the director of Jaws?
  Answer: The director of Jaws is Steven Spielberg.

  N2: Who is the director of Casino Royale?
  Answer: The director of Casino Royale is Martin Campbell.

Round 2:
  N3: Where is Steven Spielberg from?
  Answer: United States.

  N4: Where is Martin Campbell from?
  Answer: New Zealand.

Round 3:
  N5: Are United States and New Zealand the same country?
  Answer: No.

So the final answer is: No

---

Question: {question}
Graph decomposition needed: Yes.

NODES:
EDGES:
EXECUTION_ORDER:
EXECUTION:
So the final answer is:

IMPORTANT:
You MUST strictly follow this exact format:
You must always end your reasoning with:
So the final answer is: <answer>
"""

def extract_answer(text):

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

    prompt = DAG_PROMPT.format(question=question)

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
        "id": row["_id"],
        "question": question,
        "gold_answer": gold,
        "prediction": pred,
        "em": em
    }

    results.append(result)
    ems.append(em)

    time.sleep(1)

output_file = "2wiki_thinkingmethod_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)