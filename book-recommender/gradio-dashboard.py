# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dotenv>=0.9.9",
#     "gradio>=6.8.0",
#     "langchain-chroma>=1.1.0",
#     "langchain-community>=0.4.1",
#     "langchain-huggingface>=1.2.0",
#     "langchain-openai>=1.1.10",
#     "pandas>=3.0.1",
#     "sentence-transformers>=5.2.3",
# ]
# ///
import pandas as pd
import numpy as np
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

KEYWORD_WEIGHT = 0.30
TONE_WEIGHT = 0.10
MIN_QUERY_TERM_LEN = 2
ISBN_PATTERN = re.compile(r"\b97[89]\d{10}\b")
QUERY_TERM_PATTERN = re.compile(r"\b[a-z0-9]+\b")

books = pd.read_csv(BASE_DIR / "books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife = w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    str(BASE_DIR / "cover-not-found.jpg"),
    books["large_thumbnail"]
)

with open(BASE_DIR / "tagged_description.txt", encoding="utf-8") as f:
    documents = [Document(page_content=line.strip()) for line in f if line.strip()]

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    collection_name="books_fixed"
)


def _extract_isbn13(text: str) -> int | None:
    page_text = str(text).strip('"')
    first_token = page_text.split()[0] if page_text.split() else ""
    if first_token.isdigit():
        return int(first_token)

    isbn_match = ISBN_PATTERN.search(page_text)
    if isbn_match:
        return int(isbn_match.group(0))

    return None


def _extract_query_terms(query: str) -> list[str]:
    return sorted(
        {
            token
            for token in QUERY_TERM_PATTERN.findall(str(query).lower())
            if len(token) >= MIN_QUERY_TERM_LEN
        }
    )


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search_with_score(query, k = initial_top_k)
    candidate_rows = []
    for retrieval_rank, rec in enumerate(recs):
        if isinstance(rec, tuple):
            doc = rec[0]
            distance = rec[1]
        else:
            doc = rec
            distance = np.inf

        isbn13 = _extract_isbn13(getattr(doc, "page_content", ""))
        if isbn13 is None:
            continue

        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            distance_value = np.inf

        if not np.isfinite(distance_value):
            distance_value = np.inf

        candidate_rows.append(
            {
                "isbn13": isbn13,
                "distance": distance_value,
                "retrieval_rank": retrieval_rank,
            }
        )

    if not candidate_rows:
        return books.head(0)

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df = (
        candidate_df
        .groupby("isbn13", as_index=False)
        .agg(
            distance=("distance", "min"),
            retrieval_rank=("retrieval_rank", "min"),
        )
    )

    book_recs = books.merge(candidate_df, on="isbn13", how="inner")

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].copy()
    else:
        book_recs = book_recs.copy()

    if book_recs.empty:
        return books.head(0)

    safe_distance = pd.to_numeric(book_recs["distance"], errors="coerce").fillna(np.inf)
    safe_distance = safe_distance.clip(lower=0.0)
    book_recs["semantic_score"] = 1.0 / (1.0 + safe_distance)

    query_terms = _extract_query_terms(query)
    if query_terms:
        searchable_text = (
            book_recs["title"].fillna("").astype(str)
            + " "
            + book_recs["description"].fillna("").astype(str)
        ).str.lower()
        book_recs["keyword_score"] = searchable_text.apply(
            lambda text: sum(1 for term in query_terms if term in text) / len(query_terms)
        )
    else:
        book_recs["keyword_score"] = 0.0

    book_recs["final_score"] = (
        book_recs["semantic_score"] + KEYWORD_WEIGHT * book_recs["keyword_score"]
    )

    tone_column_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    tone_column = tone_column_map.get(tone)
    if tone_column:
        tone_values = pd.to_numeric(book_recs[tone_column], errors="coerce").fillna(0.0)
        book_recs["final_score"] = book_recs["final_score"] + TONE_WEIGHT * tone_values

    book_recs.sort_values(
        by=["final_score", "retrieval_rank"],
        ascending=[False, True],
        inplace=True
    )

    return book_recs.head(final_top_k)


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description_value = row.get("description", "")
        if pd.isna(description_value):
            truncated_description = "No description available."
        else:
            truncated_desc_split = str(description_value).split()
            if truncated_desc_split:
                truncated_description = " ".join(truncated_desc_split[:30]) + "..."
            else:
                truncated_description = "No description available."

        authors_value = row.get("authors", "")
        if pd.isna(authors_value):
            authors_str = "Unknown author"
        else:
            authors_split = [name.strip() for name in str(authors_value).split(";") if name.strip()]
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            elif len(authors_split) == 1:
                authors_str = authors_split[0]
            else:
                authors_str = "Unknown author"

        title_value = row.get("title", "Untitled")
        if pd.isna(title_value) or not str(title_value).strip():
            title_value = "Untitled"

        caption = f"{title_value} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tone = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book: ",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category: ", value = "All")
        tone_dropdown = gr.Dropdown(choices = tone, label = "Select an emotional tone: ", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs= output
                        )

if __name__ == "__main__":
    dashboard.launch(theme = gr.themes.Glass())
