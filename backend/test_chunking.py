from llama_index.core.node_parser import (
    SentenceSplitter,          # Method 1 - sentence aware
    TokenTextSplitter,         # Method 2 - token based  
    SemanticSplitterNodeParser # Method 3 - meaning based (best)
)
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
import re

# ─────────────────────────────────────────────
# SAMPLE LEGAL TEXT
# ─────────────────────────────────────────────
LEGAL_TEXT = """
Saradamani Kandappan vs S. Rajalakshmi & Ors
Supreme Court of India
Civil Appeal No. 6263 of 2009
Decided on: September 13, 2011

JUDGMENT

R.V. Raveendran, J.

This appeal by special leave is filed against the judgment 
dated 18.11.2008 of the Division Bench of the Madras High Court 
in Letters Patent Appeal No.60 of 2004.

FACTS OF THE CASE

The plaintiff entered into an agreement of sale with the defendant 
on 15.3.1987 in regard to a property for a consideration of 
Rs.1,05,000. The plaintiff paid an advance of Rs.5,000 on the date 
of the agreement. The balance consideration of Rs.1,00,000 was 
agreed to be paid within six months from the date of the agreement.

The plaintiff failed to pay the balance sale consideration within 
the stipulated period of six months. The defendant thereupon issued 
a notice on 22.10.1987 calling upon the plaintiff to pay the balance 
consideration and complete the sale, failing which the agreement 
would be treated as cancelled.

The plaintiff did not respond to the notice nor paid the balance 
consideration. The defendant therefore treated the agreement as 
cancelled and forfeited the advance amount.

LEGAL ISSUES

The primary legal issue before this Court is whether the time for 
performance was of the essence of the contract under Section 55 of 
the Indian Contract Act, 1872. If time is of the essence, failure 
to perform within the stipulated time would entitle the other party 
to rescind the contract.

The secondary issue concerns whether the doctrine of anticipatory 
breach applies in the present circumstances, and what remedies are 
available to the aggrieved party under Section 73 and Section 74 
of the Indian Contract Act, 1872.

HELD

The Supreme Court held that in contracts for sale of immovable 
property, time is not ordinarily of the essence unless the contract 
expressly provides so or the circumstances clearly indicate such 
intention. The mere fact that a time period is mentioned does not 
make time of the essence.

However, the Court further held that where a party fails to perform 
within a reasonable time even after the essence of time is waived, 
the other party is entitled to treat the contract as repudiated. 
The standard of reasonableness is to be judged by the facts and 
circumstances of each case.

On the question of anticipatory breach, the Court held that an 
unequivocal refusal to perform constitutes an anticipatory breach 
entitling the innocent party to immediate legal remedies. The 
aggrieved party need not wait until the date of performance to 
bring an action for damages.

CONCLUSION

The appeal was dismissed. The Court upheld the defendant's right 
to treat the contract as cancelled due to the plaintiff's failure 
to pay within a reasonable time. The forfeiture of advance amount 
was held to be valid as it represented a genuine pre-estimate of 
loss under Section 74 of the Indian Contract Act, 1872.
"""

# ─────────────────────────────────────────────
# HELPER — print chunks nicely
# ─────────────────────────────────────────────
def print_chunks(nodes, method_name):
    print("\n" + "═" * 60)
    print(f"  METHOD: {method_name}")
    print("═" * 60)
    print(f"  Total chunks: {len(nodes)}")
    
    sizes = [len(n.text) for n in nodes]
    print(f"  Avg chunk size: {sum(sizes) // len(sizes)} chars")
    print(f"  Min: {min(sizes)} chars | Max: {max(sizes)} chars")
    print("─" * 60)
    
    for i, node in enumerate(nodes):
        word_count = len(node.text.split())
        print(f"\n  CHUNK {i+1} ({word_count} words):")
        print(f"  {node.text[:250]}{'...' if len(node.text) > 250 else ''}")
    
    print("─" * 60)
    return nodes


# ─────────────────────────────────────────────
# Wrap text in LlamaIndex Document object
# This is how LlamaIndex receives any text
# ─────────────────────────────────────────────
document = Document(
    text=LEGAL_TEXT,
    metadata={
        "case_name": "Saradamani Kandappan vs S. Rajalakshmi",
        "court": "Supreme Court of India",
        "year": "2011",
        "area_of_law": "Contract Law"
    }
)


# ─────────────────────────────────────────────
# METHOD 1 — SENTENCE SPLITTER
# Splits at sentence boundaries
# Respects sentence integrity
# chunk_size is in TOKENS (not characters)
# 1 token ≈ 4 characters / 0.75 words
# ─────────────────────────────────────────────
def test_sentence_splitter():
    splitter = SentenceSplitter(
        chunk_size=128,       # tokens per chunk (~100 words)
        chunk_overlap=20,     # overlap in tokens
        paragraph_separator="\n\n",
    )
    nodes = splitter.get_nodes_from_documents([document])
    print_chunks(nodes, "1. SENTENCE SPLITTER (128 tokens, 20 overlap)")
    return nodes


# ─────────────────────────────────────────────
# METHOD 2 — TOKEN TEXT SPLITTER
# Splits purely by token count
# More precise for AI models (they think in tokens)
# ─────────────────────────────────────────────
def test_token_splitter():
    splitter = TokenTextSplitter(
        chunk_size=100,       # tokens per chunk
        chunk_overlap=20,
    )
    nodes = splitter.get_nodes_from_documents([document])
    print_chunks(nodes, "2. TOKEN SPLITTER (100 tokens, 20 overlap)")
    return nodes


# ─────────────────────────────────────────────
# METHOD 3 — LEGAL STRUCTURE AWARE (custom)
# Best for Indian court judgments
# Splits at section headings: FACTS, HELD, etc.
# Each section = one chunk (most meaningful)
# ─────────────────────────────────────────────
def test_legal_structure():
    
    # Indian judgment sections
    section_pattern = re.compile(
        r'\n(?=FACTS|HELD|JUDGMENT|CONCLUSION|LEGAL ISSUES|'
        r'BACKGROUND|ORDER|RATIO DECIDENDI|OBITER|HEADNOTES)',
        re.IGNORECASE
    )
    
    sections = section_pattern.split(LEGAL_TEXT)
    
    nodes = []
    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) < 50:   # skip tiny fragments
            continue
        
        # Extract section heading
        first_line = section.split('\n')[0].strip()
        
        # If section too long, split further with SentenceSplitter
        if len(section) > 800:
            sub_doc = Document(text=section, metadata=document.metadata)
            sub_splitter = SentenceSplitter(chunk_size=150, chunk_overlap=30)
            sub_nodes = sub_splitter.get_nodes_from_documents([sub_doc])
            for node in sub_nodes:
                node.metadata["section"] = first_line
            nodes.extend(sub_nodes)
        else:
            node = Document(
                text=section,
                metadata={**document.metadata, "section": first_line}
            )
            nodes.append(node)
    
    print_chunks(nodes, "3. LEGAL STRUCTURE AWARE (section-based)")
    return nodes


# ─────────────────────────────────────────────
# METHOD 4 — SEMANTIC SPLITTER
# Splits based on MEANING, not characters/tokens
# Uses embeddings to detect topic shifts
# Most intelligent method — needs embedding model
# ─────────────────────────────────────────────
def test_semantic_splitter():
    print("\n" + "═" * 60)
    print("  METHOD: 4. SEMANTIC SPLITTER")
    print("═" * 60)
    print("  Loading embedding model (nomic-embed-text)...")
    print("  This needs 'ollama pull nomic-embed-text' first")
    print("─" * 60)
    
    try:
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,              # sentences to group together
            breakpoint_percentile_threshold=85  # sensitivity of topic detection
                                                # higher = fewer, larger chunks
                                                # lower  = more, smaller chunks
        )
        
        nodes = splitter.get_nodes_from_documents([document])
        print_chunks(nodes, "4. SEMANTIC SPLITTER (meaning-based)")
        return nodes
        
    except Exception as e:
        print(f"\n  ⚠️  Semantic splitter failed: {e}")
        print("  → Run: ollama pull nomic-embed-text")
        print("  → Then rerun this test")
        return []


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(results):
    print("\n\n" + "═" * 60)
    print("  FINAL SUMMARY")
    print("═" * 60)
    
    methods = [
        "1. Sentence Splitter",
        "2. Token Splitter",
        "3. Legal Structure Aware",
        "4. Semantic Splitter",
    ]
    
    print(f"\n  {'Method':<30} {'Chunks':>6}  {'Avg Size':>9}")
    print("─" * 60)
    
    for method, nodes in zip(methods, results):
        if not nodes:
            print(f"  {method:<30} {'FAILED':>6}")
            continue
        sizes = [len(n.text) for n in nodes]
        avg = sum(sizes) // len(sizes)
        print(f"  {method:<30} {len(nodes):>6}  {avg:>7} chars")
    
    print("""
─────────────────────────────────────────────────────────
  RECOMMENDATION FOR LEXFORGE:

  ┌─────────────────────────────────────────────────────┐
  │  USE METHOD 3 + METHOD 1 COMBINED                   │
  │                                                     │
  │  Step 1: Split by legal section (FACTS/HELD/etc)    │
  │  Step 2: If section > 800 chars, apply              │
  │          SentenceSplitter on that section           │
  │                                                     │
  │  Result: Chunks that are both                       │
  │  → Semantically meaningful (legal section aware)    │
  │  → Right size for embedding + retrieval             │
  └─────────────────────────────────────────────────────┘

  Method 4 (Semantic) is theoretically best but:
  → Slower (runs embeddings during indexing)
  → Overkill for structured legal judgments
  → Use it only for unstructured legal articles/notes
─────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n⚖  LEXFORGE — LLAMAINDEX CHUNKING TESTER")
    print("Testing 4 chunking strategies on a real judgment\n")
    
    r1 = test_sentence_splitter()
    r2 = test_token_splitter()
    r3 = test_legal_structure()
    r4 = test_semantic_splitter()   # needs nomic-embed-text
    
    print_summary([r1, r2, r3, r4])