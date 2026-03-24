import ollama
import re

def chat_with_qwen(system_prompt, user_message, think=False):
    """
    think=False  → faster, direct answer (use this for LexForge)
    think=True   → slower, deeper reasoning (use for complex IRAC)
    """
    
    # Qwen3 specific: add /no_think to disable thinking mode
    if not think:
        user_message = user_message + " /no_think"
    
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    
    raw = response['message']['content']
    
    # Strip <think>...</think> block if present
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    
    return clean


# ── TEST 1: Basic legal question ──────────────────────────
print("=" * 50)
print("TEST 1 — Basic Legal Question")
print("=" * 50)

answer = chat_with_qwen(
    system_prompt="You are LexForge, an expert Indian legal AI assistant. Be concise and precise.",
    user_message="Explain anticipatory breach of contract under Indian law in 3 sentences."
)

print(answer)


# ── TEST 2: Structured JSON output (critical for LexForge) ─
print("\n" + "=" * 50)
print("TEST 2 — JSON Output (Issue Extraction)")
print("=" * 50)

facts = """
A software engineer signed a 2-year fixed-term contract on Jan 1 2023.
The company terminated him on Aug 15 2023 citing restructuring.
No notice or severance was given as required by the contract.
"""

json_answer = chat_with_qwen(
    system_prompt="""You are a legal AI. Always respond with valid JSON only. 
    No explanation, no markdown, just raw JSON.""",
    
    user_message=f"""Extract the legal issues from these facts as JSON.
    
Facts: {facts}

Respond in this exact format:
{{
  "issues": [
    {{"issue": "issue title", "area_of_law": "area", "priority": "high/medium/low"}},
    ...
  ]
}}"""
)

print(json_answer)

# Parse it to confirm it's valid JSON
import json
try:
    parsed = json.loads(json_answer)
    print("\n✅ Valid JSON! Issues found:")
    for i, issue in enumerate(parsed['issues'], 1):
        print(f"  {i}. {issue['issue']} ({issue['area_of_law']}) — {issue['priority']}")
except json.JSONDecodeError as e:
    print(f"❌ JSON parse failed: {e}")
    print("Raw output was:", json_answer)