import re
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()
llm_cache = {}
_client: Groq | None = None

# --------------------------
# Data structure
# --------------------------
class Product:
    def __init__(self, product_id: str, name: str, price: float):
        self.product_id = product_id
        self.name = name
        self.price = price

    def __repr__(self):
        return f"{self.product_id}: {self.name} ({self.price})"


# --------------------------
# Union-Find for transitive grouping
# --------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.parent[pb] = pa


# --------------------------
# 1. Text cleaning
# --------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_groq_client() -> Groq:
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in environment or .env file.")
    _client = Groq(api_key=key)
    return _client


def model_family_tokens(text: str) -> set[str]:
    """
    Extract model-ish alphanumeric tokens such as s23, a54, 14pro, 128gb.
    Used as a simple guardrail to reduce false merges.
    """
    tokens = re.findall(r"[a-z0-9]+", clean_text(text))
    out = set()
    for t in tokens:
        has_digit = any(ch.isdigit() for ch in t)
        if has_digit and len(t) >= 2:
            out.add(t)
    return out


def has_hebrew(text: str) -> bool:
    return re.search(r"[\u0590-\u05FF]", text) is not None


# --------------------------
# 2. LLM normalization (Groq)
# --------------------------
def llm_normalize(name: str) -> str:
    if name in llm_cache:
        return llm_cache[name]

    prompt = f"""
Normalize the following product name into a canonical English form.
- Translate Hebrew/other non-English words to English
- Preserve key attributes (model, size, version)
- Keep model families and suffixes exact (for example S23, A54, Ultra, Pro, Max)
- Normalize storage units consistently (for example 128 גיגה -> 128GB)
- Remove marketing fluff
- Keep word order consistent

Product: "{name}"

Return only the normalized name.
"""

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Normalize product names."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        norm = response.choices[0].message.content.strip()
        llm_cache[name] = norm
        return norm

    except Exception:
        return name  # fallback


# --------------------------
# 3. Similarity check
# --------------------------
def is_same_product(p1: Product, p2: Product, threshold=79) -> bool:
    n1 = clean_text(llm_normalize(p1.name))
    n2 = clean_text(llm_normalize(p2.name))

    # Guardrail 1: strong tier words usually indicate different SKUs.
    tier_words = ("ultra", "pro", "max", "plus")
    for w in tier_words:
        if (w in n1) != (w in n2):
            return False

    # Guardrail 2: if both have model-family tokens and they do not overlap, reject.
    m1 = model_family_tokens(n1)
    m2 = model_family_tokens(n2)
    # Relax this rule for Hebrew/transliterated pairs where tokenization may differ.
    if m1 and m2 and not (m1 & m2) and not (has_hebrew(p1.name) or has_hebrew(p2.name)):
        return False

    score = fuzz.token_sort_ratio(n1, n2)
    return score >= threshold


# --------------------------
# 4. Group products transitively
# --------------------------
def group_products(products: List[Product], threshold: int = 79) -> List[List[Product]]:
    n = len(products)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if is_same_product(products[i], products[j], threshold=threshold):
                uf.union(i, j)

    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(products[idx])

    return list(clusters.values())


# --------------------------
# 5. Pick cheapest per group
# --------------------------
def get_cheapest(groups: List[List[Product]]) -> List[Product]:
    return [min(g, key=lambda p: p.price) for g in groups]


def run(input_csv: Path, output_csv: Path, threshold: int = 79) -> None:
    df = pd.read_csv(input_csv)
    for col in ("product_id", "title", "price"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    products = [
        Product(str(r.product_id), str(r.title).strip(), float(r.price))
        for r in df.itertuples(index=False)
    ]

    groups = group_products(products, threshold=threshold)

    # Map each product_id to the cluster's cheapest price.
    lowest_by_product: Dict[str, float] = {}
    for g in groups:
        lowest = min(p.price for p in g)
        for p in g:
            lowest_by_product[p.product_id] = lowest

    out_df = df.copy()
    out_df["lowest_price_in_group"] = out_df["product_id"].astype(str).map(lowest_by_product)
    out_df.to_csv(output_csv, index=False)

    print(f"Wrote {output_csv}")
    print(out_df[["product_id", "title", "price", "lowest_price_in_group"]].to_string(index=False))


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    input_csv = Path("data/sample_products.csv")
    output_csv = Path("output/deduped_simple.csv")
    threshold = 79
    run(input_csv, output_csv, threshold=threshold)