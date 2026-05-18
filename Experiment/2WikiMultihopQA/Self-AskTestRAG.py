import time
import re
import string
import json
from openai import OpenAI
from rank_bm25 import BM25Okapi

deepseek_api_key = ""
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def ask_llm(prompt, stop=None):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    text = response.choices[0].message.content
    return text

def extract_answer(text):
    text = text.replace('*', '').replace('_', '')

    match = re.search(
        r"So the final answer is:\s*(.*?)(?:\.|\n|$)",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]

        if ':' in last_line:
            return last_line.split(':',1)[1].strip()
        return last_line
    return text.strip()

def extract_question(text):
    last_line = text.strip().split("\n")[-1]
    last_line = last_line.replace('*','').replace('_','')
    if "Follow up:" in last_line:
        after_colon = last_line.split(":",1)[1].strip()
        return after_colon
    return None

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def retrieve_bm25(query, topk=5):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [corpus[i] for i in ranked_idx]

def build_context(docs):
    return "\n\n".join([f"{d['title']}: {d['text']}" for d in docs])

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

def selfask_rag(question, examples, bm25, corpus):
    # prompt
    cur_prompt = examples + "\nQuestion: " + question + "\nAre follow up questions needed here:"

    # Get followed up
    ret_text = ask_llm(cur_prompt, stop="Intermediate answer:")
    print("============ret_text1================")
    print(ret_text)

    # Check followed up
    while "Follow up:" in ret_text.split("\n")[-1]:
        # Get child answer
        sub_question = extract_question(ret_text)
        print("===========sub_question=================")
        print(sub_question)
        if sub_question is None:
            break

        # RAG
        docs = retrieve_bm25(sub_question, topk=5)

        print("\nRetrieved Documents Titles:")
        for d in docs:
            print("-", d['title'])

        if docs:
            context = build_context(docs)
            read_prompt = (
                f"Based on the following documents, answer the question in one concise sentence.\n\n"
                f"Documents:\n{context}\n\n"
                f"Question: {sub_question}\n"
                f"Answer:"
            )
            external_answer = ask_llm(read_prompt).strip()
            intermediate_text = f"Intermediate text: {external_answer}."
        else:
            # no answer, directly use llm
            intermediate_text = "Intermediate text:"

        # 5. intermediate answer
        cur_prompt += (ret_text + "\n" + intermediate_text + "\n" + "Are follow up questions needed here:")
                      # "IMPORTANT: When writing So the final answer is:, only output the final concise answer."
        print(intermediate_text)

        # use llm again
        ret_text = ask_llm(cur_prompt, stop=["Intermediate answer:", "Follow up:", "So the final answer is:"])
        print("============ret_text2================")
        print(ret_text)

    # Final answer output
    if "So the final answer is:" not in ret_text:
        cur_prompt += ret_text + "\nSo the final answer is:"
        ret_text = ask_llm(cur_prompt, stop=None)

    final_answer = extract_answer(ret_text)
    return final_answer, ret_text

with open("2wiki_sample_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("rag_paragraph_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
tokenized_corpus = [tokenize(text) for text in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

selfask_examples = """

Question: Who lived longer, Muhammad Ali or Alan Turing? 
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born? A
re follow up questions needed here: Yes.
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
"""

results = []
ems = []

for idx, row in enumerate(data):
    question = row["question"]
    gold = row["answer"]
    print(f"\n==== {idx} ====")
    print("Question:", question)

    try:
        pred, full_text = selfask_rag(question, selfask_examples, bm25, corpus)
        print("============Full Reasoning================")
        print("Full Reasoning:\n", full_text)
        print("============Predicted Answer================")
        print("Predicted Answer:", pred)
    except Exception as e:
        print("Error:", e)
        pred = ""

    em = compute_em(pred, gold)
    print("EM:", em)

    results.append({
        "id": row["_id"],
        "question": question,
        "gold_answer": gold,
        "prediction": pred,
        "em": em
    })
    ems.append(em)
    time.sleep(1)

output_file = "2wiki_selfaskRAG_results.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)