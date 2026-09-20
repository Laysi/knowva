"""Three turns of one conversation: which pages each turn retrieves, and its cost.

Since v0.4.1 the layout of the context blocks depends on `previousSources`: the
sources the previous answer cited, echoed back unchanged. Without it the backend
falls back to ranking order and the prompt cache sees a fresh prefix every turn.
"""
import json, time, requests

URL = "http://127.0.0.1:8080/api/chat"

def turn(messages, label, previous_sources):
    start = time.time(); first = None; answer = []; sources = []
    payload = {"messages": messages, "previousSources": previous_sources}
    with requests.post(URL, json=payload, stream=True, timeout=1800) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "text-delta":
                first = first or time.time() - start
                answer.append(event["text"])
            elif event.get("type") == "sources":
                sources = event["sources"]
            elif event.get("type") == "error":
                print(f"{label}: ERROR {event['message']}", flush=True)
                return None, []
    pages = [s.get("pageNumber") for s in sources]
    units = [s.get("unitId") for s in sources]
    print(f"{label}: first={first:.0f}s total={time.time()-start:.0f}s pages={pages} units={units}", flush=True)
    return "".join(answer), sources

questions = ["這份報告提到哪些工業物聯網的重點?",
             "那資安的部分呢?",
             "有提到哪些通訊協定或平台?"]
history = []
previous = []
for index, question in enumerate(questions, 1):
    history.append({"role": "user", "content": question})
    answer, sources = turn(history, f"turn {index}", previous)
    if answer is None:
        break
    if previous:
        carried = [s["unitId"] for s in sources if s["unitId"] in {p["unitId"] for p in previous}]
        prefix = 0
        for new, old in zip(sources, previous):
            if new["unitId"] != old["unitId"]:
                break
            prefix += 1
        print(f"  carried={len(carried)}/{len(sources)} stable_prefix={prefix} blocks", flush=True)
    previous = sources
    history.append({"role": "assistant", "content": answer})
print("THREE_TURN_DONE", flush=True)
