import time
import re
import string
import json
from openai import OpenAI
from rank_bm25 import BM25Okapi

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

cot_rag_prompt_template = """
Use the following context to answer the question.

Context:
{context}

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
Let's think step by step.
So the final answer is:

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

def retrieve_bm25(query, topk=5):

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:topk]

    return [corpus[i] for i in ranked_idx]

def build_context(docs):

    context = ""

    for d in docs:
        context += f"{d['title']}: {d['text']}\n\n"

    return context

def tokenize(text):
    return re.findall(r"\w+", text.lower())

with open("musique_sample_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("rag_paragraph_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_texts = [
    doc["title"] + " " + doc["text"]
    for doc in corpus
]

tokenized_corpus = [
    tokenize(text)
    for text in corpus_texts
]

bm25 = BM25Okapi(tokenized_corpus)

results = []
ems = []

for idx, row in enumerate(data):

    question = row["question"]
    gold = row["answer"]

    print(f"\n==== {idx} ====")
    print("Question:", question)

    docs = retrieve_bm25(question, topk=5)

    print("\nRetrieved Documents:")
    for d in docs:
        print("-", d["title"])

    context = build_context(docs)

    prompt = cot_rag_prompt_template.format(
        context=context,
        question=question
    )

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

output_file = "musique_cotRAG_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)