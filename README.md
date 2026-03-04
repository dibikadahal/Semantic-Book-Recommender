# Semantic Book Recommender
An AI-powered book recommendation system using LLMs, vector search, zero-shot classification, and sentiment analysis with an interactive Gradio dashboard.
<img width="2473" height="1068" alt="image" src="https://github.com/user-attachments/assets/0db91e49-2b53-41f0-9902-0bcd8b79cbe8" />
<img width="2488" height="1168" alt="image" src="https://github.com/user-attachments/assets/4aff41f0-d6c3-424a-96bc-85c88e618039" />
<img width="2295" height="811" alt="image" src="https://github.com/user-attachments/assets/21bc7fdf-4fc6-4746-a359-75a2e90e87d1" />

## 📖 Overview
The Semantic Book Recommender is a full-stack machine learning project that recommends books based on the meaning of a user's query — not just keyword matching. It uses document embeddings, vector similarity search, zero-shot text classification, and fine-tuned sentiment analysis to power an intelligent, filterable book recommendation dashboard.

Tech Stack summary:
- 🧠 Brain: Hugging Face models (transformers)
- 🎨 Face (UI): Gradio dashboard
- 🗄️ Memory: ChromaDB vector database
- 🔗 Framework: LangChain

## ✨ Features
- Semantic Search — Find books similar in theme and writing style using vector embeddings and similarity
- Zero-Shot Genre Classification — Automatically classify books as Fiction, Nonfiction, Children's Fiction, or Children's Nonfiction without labeled training data
- Sentiment / Emotion Analysis — Fine-tuned emotion detection across 7 categories (joy, sadness, anger, fear, disgust, surprise, neutral) per book description
- Interactive Gradio Dashboard — Filter and sort recommendations by genre and emotional tone
- Data Cleaning Pipeline — Handling missing values, short descriptions, long-tail category normalization, and subtitle merging

## 🗂️ Project Structure
```text
book-recommender/
│
├── 📓 data-exploration.ipynb        # EDA, data cleaning, missing value analysis
├── 📓 vector-search.ipynb           # Document embeddings + ChromaDB vector search
├── 📓 text-classification.ipynb     # Zero-shot genre classification
├── 📓 sentiment-analysis.ipynb      # Emotion classification per book
│
├── 🐍 gradio-dashboard.py           # Final Gradio UI — main app
│
├── 📄 books_cleaned.csv             # Cleaned dataset (post EDA)
├── 📄 books_with_categories.csv     # Dataset after genre classification
├── 📄 books_with_emotions.csv       # Final dataset with emotion scores
├── 📄 tagged_description.txt        # ISBN-tagged descriptions for vector DB
│
├── 🖼️  cover-not-found.jpg           # Fallback image for missing book covers
├── 📄 requirements.txt              # All dependencies
└── 📄 .env                          # API keys (not committed — create manually)
```

## 🧱 Pipeline Architecture
```text
Kaggle Dataset (7K Books)
        │
        ▼
┌──────────────────────┐
│   data-exploration   │  ← pandas, matplotlib, seaborn
│   - Handle NaN       │    Heatmap of missing values
│   - Filter short     │    Correlation analysis
│     descriptions     │    Word count filtering (≥25 words)
│   - Merge title +    │
│     subtitle         │
└────────┬─────────────┘
         │  books_cleaned.csv
         ▼
┌────────────────────────────┐
│      vector-search         │  ← LangChain + HuggingFace Embeddings
│  - Tag descriptions w/ISBN │    sentence-transformers/all-MiniLM-L6-v2
│  - Build ChromaDB index    │    ChromaDB collection: "books_fixed"
│  - similarity_search()     │
└────────┬───────────────────┘
         │
         ▼
┌──────────────────────────────┐
│    text-classification       │  ← facebook/bart-large-mnli
│  - Zero-shot classification  │    Fiction / Nonfiction / Children's
│  - ~78% accuracy on test set │    Missing categories filled by LLM
└────────┬─────────────────────┘
         │  books_with_categories.csv
         ▼
┌──────────────────────────────────┐
│      sentiment-analysis          │  ← j-hartmann/emotion-english-distilroberta-base
│  - Per-sentence classification   │    7 emotion categories
│  - Max score aggregation         │    Merged back into main DataFrame
└────────┬─────────────────────────┘
         │  books_with_emotions.csv
         ▼
┌──────────────────────┐
│   gradio-dashboard   │  ← Filter by category + sort by emotional tone
│  - Book covers       │    16 recommendations displayed as a gallery
│  - Truncated desc    │
│  - Author formatting │
└──────────────────────┘
```

## ⚙️ Setup & Installation
1. Clone the Repository
```bash
   git clone https://github.com/dibikadahal/Semantic-Book-Recommender.git
   cd semantic-book-recommender
```

2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Configure API keys
Create a .env file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```
- Get your OpenAI key at [platform.openai.com](https://platform.openai.com/)
- Get your HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

5. Download the Dataset
The dataset is fetched automatically via kagglehub inside the notebooks. Make sure your Kaggle credentials are configured, then the first cell of data-exploration.ipynb handles the download:
```python
import kagglehub
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
```

## 🚀 Running the Project
Run the notebooks in order — each one produces a file that the next step depends on.

Step1 - Data Exploration and Cleaning
```
Open:   data-exploration.ipynb
Output: books_cleaned.csv
```

Step2 - Build Vector Search Index
```
Open:   vector-search.ipynb
Output: tagged_description.txt + ChromaDB collection (books_fixed)
```

Step3 - Genre Classification
```
Open:   text-classification.ipynb
Output: books_with_categories.csv
```

Step4 - Emotion Analysis
```
Open:   sentiment-analysis.ipynb
Output: books_with_emotions.csv
```

Step5 -   Launch the dashboard
```
python gradio-dashboard.py
```
Open the URL shown in the terminal.


## 🖥️ GPU Compatibility Notes
⚠️ This is a known issue for users with newer NVIDIA GPUs.

This project was developed and tested on two machines, and a significant hardware compatibility difference was discovered:
| GPU | CUDA Support | Status |
|-----|--------------|---------|
| NVIDIA RTX 4060 | ✅ Fully supported by stable PyTorch | Works perfectly |
| NVIDIA RTX 5060 | ❌ Not supported by stable PyTorch | Falls back to CPU silently|

The Problem:
The RTX 5060 uses NVIDIA's Blackwell architecture (SM 1.20), which is not yet included in the stable PyTorch release. When running on an RTX 5060, you will see this warning:
```
UserWarning: NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120
is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities:
sm_37, sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90.
If you want to use the NVIDIA GeForce RTX 5060 Laptop GPU with PyTorch,
please check the instructions at https://pytorch.org/get-started/locally/
```
The model will still run, but falls back silently to CPU, making the classification notebooks significantly slower.

Fix - Install PyTorch Nightly for RTX5060
Uninstall the current torch version and install the nightly build with CUDA 12.8 support:
```bash
pip uninstall torch torchvision torchaudio -y

pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```
Verify the GPU is now detected:
```python
import torch
print(torch.cuda.is_available())       # Should print: True
print(torch.cuda.get_device_name(0))   # Should print: NVIDIA GeForce RTX 5060
```
Alternative — Run on CPU Only
If you prefer not to install nightly builds, change the device argument in the pipeline calls inside the notebooks:
```python
# Change this:
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cuda")

# To this:
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
```
The project is fully functional on CPU — just expect the classification steps to take longer.


## 🎛️ Dashboard Usage
Once the Gradio app is running:
1. Enter a description — Type any book theme or query (e.g., "a young wizard discovering his magical powers" or "historical fiction set during World War II")
2. Select a category — Optionally filter by Fiction, Nonfiction, Children's Fiction, Children's Nonfiction, or All
3. Select an emotional tone — Sort results by Happy, Sad, Surprising, Angry, Suspenseful, or All
4. Click "Find recommendations" — A gallery of 16 recommended books appears with covers, titles, authors, and truncated descriptions

## 🤝Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is for educational purposes. Dataset sourced from [Kaggle — 7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) by Dylan Castillo.

## 🙏 Acknowledgements
- [LangChain](https://langchain.com/) for the powerful LLM orchestration framework
- [Hugging Face](https://huggingface.co/) for open-source models and the Transformers library
- [Gradio](https://gradio.app/) for making ML dashboards easy to build
- [ChromaDB](https://www.trychroma.com/) for the lightweight local vector database
- Dataset by [Dylan Castillo](https://www.kaggle.com/dylanjcastillo) on Kaggle
