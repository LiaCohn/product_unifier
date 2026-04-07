
import pytest
from main import (
    clean_text,
    model_family_tokens,
    is_same_product,
    group_products,
    Product,
    llm_normalize,
    llm_cache,
)


# --------------------------
# Mock LLM
# --------------------------
def mock_llm(name: str) -> str:
    mapping = {
        "סמסונג גלקסי 23": "Samsung Galaxy S23",
        "Samsung S23": "Samsung Galaxy S23",
        "אייפון 14": "iPhone 14",
        "Apple iPhone 14": "iPhone 14",
    }
    return mapping.get(name, name)


# --------------------------
# clean_text
# --------------------------
def test_clean_text():
    assert clean_text("Samsung!!! Galaxy S23") == "samsung galaxy s23"
    assert clean_text("   Hello   World ") == "hello world"


# --------------------------
# model_family_tokens
# --------------------------
def test_model_family_tokens():
    tokens = model_family_tokens("Samsung Galaxy S23 128GB")
    assert "s23" in tokens
    assert "128gb" in tokens


# --------------------------
# is_same_product
# --------------------------
def test_is_same_product_true(monkeypatch):
    monkeypatch.setattr("main.llm_normalize", mock_llm)

    p1 = Product("1", "סמסונג גלקסי 23", 2800)
    p2 = Product("2", "Samsung S23", 3000)

    assert is_same_product(p1, p2)


def test_is_same_product_false(monkeypatch):
    monkeypatch.setattr("main.llm_normalize", mock_llm)

    p1 = Product("1", "Samsung Galaxy S23", 3000)
    p2 = Product("2", "Samsung Galaxy S22", 2800)

    assert not is_same_product(p1, p2)


def test_tier_word_guardrail_false(monkeypatch):
    monkeypatch.setattr("main.llm_normalize", lambda x: x)

    p1 = Product("1", "Samsung Galaxy S23", 3000)
    p2 = Product("2", "Samsung Galaxy S23 Ultra", 4200)

    assert not is_same_product(p1, p2)


# --------------------------
# group_products
# --------------------------
def test_group_products(monkeypatch):
    monkeypatch.setattr("main.llm_normalize", mock_llm)

    products = [
        Product("1", "Samsung Galaxy S23", 3000),
        Product("2", "סמסונג גלקסי 23", 2800),
        Product("3", "iPhone 14", 3300),
        Product("4", "אייפון 14", 3200),
    ]

    groups = group_products(products)

    # Expect 2 groups: Samsung + iPhone
    assert len(groups) == 2

    # Optional: verify sizes
    sizes = sorted(len(g) for g in groups)
    assert sizes == [2, 2]


# --------------------------
# cheapest selection (integration style)
# --------------------------
def test_cheapest_in_group(monkeypatch):
    monkeypatch.setattr("main.llm_normalize", mock_llm)

    products = [
        Product("1", "Samsung Galaxy S23", 3000),
        Product("2", "סמסונג גלקסי 23", 2800),
    ]

    groups = group_products(products)

    cheapest = min(groups[0], key=lambda p: p.price)
    assert cheapest.price == 2800


# --------------------------
# llm cache behavior
# --------------------------
def test_llm_normalize_uses_cache(monkeypatch):
    llm_cache.clear()
    calls = {"n": 0}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            class _Msg:
                content = "Samsung Galaxy S23"
            class _Choice:
                message = _Msg()
            class _Resp:
                choices = [_Choice()]
            return _Resp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr("main.get_groq_client", lambda: _FakeClient())

    a = llm_normalize("סמסונג גלקסי 23")
    b = llm_normalize("סמסונג גלקסי 23")

    assert a == "Samsung Galaxy S23"
    assert b == "Samsung Galaxy S23"
    assert calls["n"] == 1