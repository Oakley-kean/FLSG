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

with open("HotpotQA_sample_100.json", "r", encoding="utf-8") as f:
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
        "id": row["_id"],
        "question": question,
        "gold_answer": gold,
        "prediction": pred,
        "em": em
    }

    results.append(result)
    ems.append(em)

    time.sleep(1)

output_file = "hotpot_cotRAG_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)