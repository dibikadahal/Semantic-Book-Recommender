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
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife = w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

with open("tagged_description.txt", encoding="utf-8") as f:
    documents = [Document(page_content=line.strip()) for line in f if line.strip()]

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    collection_name="books_fixed"
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search_with_score(query, k = initial_top_k)
    isbn_pattern = re.compile(r"\b97[89]\d{10}\b")
    books_list = []
    for rec in recs:
        doc = rec[0] if isinstance(rec, tuple) else rec
        page_text = doc.page_content.strip('"')
        first_token = page_text.split()[0] if page_text.split() else ""

        if first_token.isdigit():
            books_list.append(int(first_token))
            continue

        isbn_match = isbn_pattern.search(page_text)
        if isbn_match:
            books_list.append(int(isbn_match.group(0)))

    if not books_list:
        return books.head(0)

    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)


    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)


    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
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
