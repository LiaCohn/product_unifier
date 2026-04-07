# Product Dedup 


The script attempts to solve the task:
- find duplicate products with inconsistent names (Hebrew/English variants),
- group duplicates transitively,
- and output the lowest price for each product's group.

## How it works

1. Read input CSV with required columns: `product_id`, `title`, `price`.
2. Normalize each title with Groq LLM (`llama-3.1-8b-instant`), with caching per raw title.
3. Compare product pairs using:
   - fuzzy similarity on normalized names,
   - guardrails to avoid obvious false merges (for example `Ultra/Pro/Max/Plus` mismatches).
4. Build duplicate groups with union-find (transitive closure).
5. For each group, compute `lowest_price_in_group`.
6. Write output CSV.

## Install

```bash
cd product_unifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create `.env` (or export in shell):

```bash
GROQ_API_KEY=gsk_...
```

`GROQ_API_KEY` is required to run `main.py`.
You can create/get a key from the Groq Console: [https://console.groq.com/keys](https://console.groq.com/keys)

## Run

`main.py` uses fixed paths and threshold inside the script:
- input: `data/sample_products.csv`
- output: `output/deduped_simple.csv`
- threshold: `79`

Run:

```bash
python main.py
```

## Output

The output file is:

- `output/deduped_simple.csv`

Main added column:

- `lowest_price_in_group`

This value is the minimum price among all products in the matched duplicate cluster.


## Limitations

This is not a perfect solution and intentionally keeps the logic simple.

The approach relies on:
LLM normalization quality
fuzzy similarity thresholds
It is more effective for structured product names (e.g. electronics)
It may struggle with:
- very short or ambiguous titles
- non-standard naming
- domain-specific attributes (e.g. furniture dimensions, fashion sizes)

## Possible Improvements
- Domain-specific tuning (per category: electronics, furniture, fashion, etc.)
- Better structured extraction (attributes like size, color, material)
- Replacing pairwise matching with scalable clustering approaches
- Using embedding-based similarity instead of (or alongside) fuzzy matching
- Learning-based approaches (supervised duplicate detection)

## Summary

This solution demonstrates a practical hybrid approach:

LLM for normalization
heuristics for control
clustering for grouping

The focus is on building a simple, explainable pipeline that can be extended and adapted to different domains.