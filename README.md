# NLP Preprocessing

**A collection of three NLP assignments implementing preprocessing, embeddings, and seq2seq summarization.**

---

## Assignment 1.1: NLP Preprocessing Basics
### Core Features
- **Implemented Techniques**:
  - Tokenization
  - Stemming
  - Lemmatization
  - Named Entity Recognition (NER)
- **Comparison Table**: 10+ stem vs. lemma examples
- **Web Interface**: FastAPI backend + HTML/Jinja2 frontend
- **Alternative UI**: Gradio interface
- **API Endpoints**:
  - `POST /api/process` - Returns all preprocessing results
  - `GET /api/compare` - Returns stem/lemma comparison table

### Technologies Used
NLTK | FastAPI | HTML/CSS | Gradio


---

## Assignment 1.2: Word Embeddings & Visualization
### Key Components
- **Embedding Methods**:
  - TF-IDF (scikit-learn)
  - PCA Visualization (2D)
- **Interactive Features**:
  - Word similarity comparison
  - Dimensionality reduction control
- **Web Interface**: Gradio UI

### Sample Visualization
TF-IDF to PCA reduction example

from sklearn.decomposition import PCA
tfidf_matrix = vectorizer.fit_transform(corpus)
pca = PCA(n_components=2)
vis_coords = pca.fit_transform(tfidf_matrix.toarray())

## Assignment 1.3: Seq2Seq Text Summarization with LSTM

- **Synthetic Dataset**: 200 auto-generated news articles (title + summary pairs)
- **Seq2Seq Architecture**: LSTM-based encoder-decoder
- **Training**: Teacher forcing + early stopping
- **Evaluation**: BLEU-4 score implementation
  


## 📚 Documentation
### Key Implementation Choices
1. **Stemming vs Lemmatization**
   - Porter Stemmer for speed
   - WordNet Lemmatizer for accuracy
   - POS-aware lemmatization

2. **TF-IDF Optimization**
   - Sublinear TF scaling
   - English stopword removal
   - Custom n-gram range (1-3)

3. **Seq2Seq Enhancements**
   - Synthetic data generation
   - Encoder Decoder model creation
   - Evaluation

## Conclusion
## NLP Assignment Suite

**Three NLP projects covering preprocessing, embeddings, and text summarization**  
*Implemented with Python, NLTK, TensorFlow, and interactive web interfaces*

---

## Quick Start

### 1. Clone Repository
git clone https://github.com/navin123456789/NLP-Preprocessing-Basics.git
cd nlp-assignment-suite
### 2. Install Dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt wordnet averaged_perceptron_tagger

## nltk==3.8.1
fastapi==0.109.2
uvicorn==0.27.1
scikit-learn==1.4.0
gradio==4.14.0
tensorflow==2.15.0 Dependencies

## This README features:
1.All Projects in One Place

2.Zero Guesswork Setup

3.Technical Deep Dives

4.Dependency Made Simple



