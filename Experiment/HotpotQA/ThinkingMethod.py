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

DAG_PROMPT = """
Analyze the following multi-hop question and decompose it into a reasoning graph.
Rules:
1. Each node is a simple single-hop sub-question
2. A node depends on another if it needs that node's answer to form its question
3. Nodes with no dependencies can be answered independently

Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Graph decomposition needed: Yes.

NODES:
- N1: Who worked with Modern Records?
- N2: Is [N1_answer] an American musician, singer, actor, comedian, and songwriter, born on December 5, 1932?

EDGES:
- N2 depends on N1

EXECUTION_ORDER:
- Round 1: N1
- Round 2: N2

EXECUTION:
Round 1:
  N1: Who worked with Modern Records?
  Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker.

Round 2:
  N2: Is Etta James, Little Richard, Joe Houston, Ike and Tina Turner or John Lee Hooker an American musician, singer, actor, comedian, and songwriter, born on December 5, 1932?
  Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter.

So the final answer is: Little Richard

---

Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
Graph decomposition needed: Yes.

NODES:
- N1: What jobs did Chinua Achebe have?
- N2: What jobs did Rachel Carson have?
- N3: Who had more jobs, [N1_answer] or [N2_answer]?

EDGES:
- N3 depends on N1
- N3 depends on N2

EXECUTION_ORDER:
- Round 1 (parallel): N1, N2
- Round 2: N3

EXECUTION:
Round 1:
  N1: What jobs did Chinua Achebe have?
  Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs.

  N2: What jobs did Rachel Carson have?
  Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs.

Round 2:
  N3: Who had more jobs, Chinua Achebe (4 jobs) or Rachel Carson (3 jobs)?
  Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so Chinua Achebe had more diverse jobs.

So the final answer is: Chinua Achebe

---

Question: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
Graph decomposition needed: Yes.

NODES:
- N1: Which American rapper is featured in Remember Me Ballin', a CD single by Indo G?
- N2: In which year was [N1_answer] born?

EDGES:
- N2 depends on N1

EXECUTION_ORDER:
- Round 1: N1
- Round 2: N2

EXECUTION:
Round 1:
  N1: Which American rapper is featured in Remember Me Ballin', a CD single by Indo G?
  Intermediate answer: Gangsta Boo.

Round 2:
  N2: In which year was Gangsta Boo born?
  Intermediate answer: Gangsta Boo was born on August 7, 1979, so the answer is 1979.

So the final answer is: 1979

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

with open("HotpotQA_sample_100.json", "r", encoding="utf-8") as f:
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

output_file = "hotpot_thinkingmethod_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)