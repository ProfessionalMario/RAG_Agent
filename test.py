from agent.reasoning_agent import build_prompt, call_llm

prompt = build_prompt(column_data, retrieved_docs)

response = call_llm(prompt)

print(response)