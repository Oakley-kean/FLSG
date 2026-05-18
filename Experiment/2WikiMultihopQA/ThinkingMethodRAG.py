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

DAG_PLAN_PROMPT = """
Analyze the following multi-hop question and decompose it into a reasoning graph.
Rules:
1. Each node is a simple single-hop sub-question
2. A node depends on another if it needs that node's answer
3. Independent nodes can be answered in parallel
4. Use [Nx_answer] as placeholder when a node needs another node's answer

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

---

Question: {question}
Graph decomposition needed: Yes.

NODES:
EDGES:
EXECUTION_ORDER:

IMPORTANT: Output ONLY the NODES, EDGES, and EXECUTION_ORDER sections.
Do NOT execute or answer anything.
"""

NODE_ANSWER_PROMPT = """
Answer the following simple question using the retrieved knowledge.
Be concise and direct, output only the answer.

Retrieved Knowledge:
{context}

Question: {sub_question}

Answer:
"""

AGGREGATE_PROMPT = """
You are given a complex question and answers to all sub-questions.
Use the sub-question answers as evidence to give the final answer.

Original Question: {question}

Sub-question answers:
{sub_qa_pairs}

Reason step by step and give the final answer.
So the final answer is:

IMPORTANT: Always end with exactly: So the final answer is: <answer>
"""

def parse_dag(plan_text):
    """
    Get nodes、edges、execution_order
    {
        "nodes": {"N1": "question text", "N2": "..."},
        "order": [["N1","N2"], ["N3"], ["N4","N5"]]
    }
    """
    nodes = {}
    order = []

    lines = plan_text.strip().split("\n")
    section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("NODES:"):
            section = "nodes"
        elif line.startswith("EDGES:"):
            section = "edges"
        elif line.startswith("EXECUTION_ORDER:"):
            section = "order"

        elif section == "nodes" and line.startswith("-"):
            # "- N1: How old was Muhammad Ali..."
            match = re.match(r"-\s*(N\d+):\s*(.+)", line)
            if match:
                node_id = match.group(1)
                question = match.group(2).strip()
                nodes[node_id] = question

        elif section == "order" and line.startswith("-"):
            # "- Round 1 (parallel): N1, N2"
            # "- Round 2: N3"
            match = re.match(r"-\s*Round\s*\d+.*?:\s*(.+)", line)
            if match:
                node_list = [n.strip() for n in match.group(1).split(",")]
                order.append(node_list)

    return {"nodes": nodes, "order": order}


def fill_placeholders(question_text, node_answers):
    """
    Let [N1_answer] to real answer
    """
    for node_id, answer in node_answers.items():
        placeholder = f"[{node_id}_answer]"
        question_text = question_text.replace(placeholder, answer)
    return question_text


def format_sub_qa(node_questions, node_answers):
    lines = []
    for node_id, question in node_questions.items():
        answer = node_answers.get(node_id, "N/A")
        lines.append(f"{node_id}: {question}")
        lines.append(f"Answer: {answer}")
        lines.append("")
    return "\n".join(lines)

def execute_dag_rag(question, plan_text):
    """
    Analysis DAG, As Round do BM25+llm
    """
    dag = parse_dag(plan_text)
    nodes = dag["nodes"]
    order = dag["order"]

    node_answers = {}  # save every node

    for round_idx, round_nodes in enumerate(order):
        print(f"\n  --- Round {round_idx + 1}: {round_nodes} ---")

        for node_id in round_nodes:
            if node_id not in nodes:
                continue

            # replace N as real answer
            raw_question = nodes[node_id]
            filled_question = fill_placeholders(raw_question, node_answers)
            print(f"  [{node_id}] Query: {filled_question}")

            # BM25
            docs = retrieve_bm25(filled_question, topk=5)
            context = build_context(docs)
            print(f"  [{node_id}] Retrieved: {[d['title'] for d in docs]}")

            # LLM answer
            prompt = NODE_ANSWER_PROMPT.format(
                context=context,
                sub_question=filled_question
            )
            answer = ask_llm(prompt)
            # only first line as answer
            answer = answer.strip().split("\n")[0].strip()
            print(f"  [{node_id}] Answer: {answer}")

            # save answer to dict
            node_answers[node_id] = answer

        time.sleep(0.5)

    return node_answers

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

def tokenize(text):
    return re.findall(r"\w+", text.lower())

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

with open("2wiki_sample_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("rag_paragraph_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_texts = [doc["title"] + " " + doc["text"] for doc in corpus]
tokenized_corpus = [tokenize(text) for text in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

results = []
ems = []

for idx, row in enumerate(data):
    question = row["question"]
    gold = row["answer"]

    print(f"\n==== {idx} ====")
    print("Question:", question)

    try:
        # Step 1: DAG
        print(f"\n--- Step 1: DAG Planning ---")
        plan_prompt = DAG_PLAN_PROMPT.format(question=question)
        plan_text = ask_llm(plan_prompt)
        print("Plan Output:\n", plan_text)

        # Step 2: Node do retrievers, and answer
        print(f"\n--- Step 2: Node Execution ---")
        node_answers = execute_dag_rag(question, plan_text)

        # Step 3: get  final answer
        print(f"\n--- Step 3: Aggregation ---")
        dag = parse_dag(plan_text)
        sub_qa_text = format_sub_qa(dag["nodes"], node_answers)
        print("Sub QA pairs fed to aggregator:\n", sub_qa_text)

        agg_prompt = AGGREGATE_PROMPT.format(
            question=question,
            sub_qa_pairs=sub_qa_text
        )
        final_output = ask_llm(agg_prompt)
        print("Final output:\n", final_output)

        pred = extract_answer(final_output)

    except Exception as e:
        print("Error:", e)
        pred = ""

    print("Predicted Answer:", pred)
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

output_file = "2wiki_thinkingmethodRAG2_results.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

accuracy = sum(ems) / len(ems)
print("\nFinal EM Accuracy:", accuracy)