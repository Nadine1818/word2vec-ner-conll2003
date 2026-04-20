# Word2Vec Embeddings + NER on CoNLL-2003

End-to-end NLP notebook that:
1) trains **Word2Vec-style embeddings from scratch** (negative sampling), and
2) applies **Named Entity Recognition (NER)** using two approaches:
	 - a simple **Neural Network** classifier, and
	 - a **Hidden Markov Model (HMM)** with **Viterbi decoding**.

The work uses the **CoNLL-2003** dataset via Hugging Face Datasets.

## Why this project (CV summary)
- Built a full NLP pipeline: dataset loading → preprocessing → embedding training → sequence labeling.
- Implemented core ML primitives manually (e.g., sampling table, negative sampling, training loop) to demonstrate fundamentals.
- Compared probabilistic sequence modeling (HMM) vs. neural modeling for NER.

## What’s inside
- `word2vec_ner_conll2003.ipynb` — main notebook
	- Load and preprocess CoNLL-2003 tokens
	- Define hyperparameters
	- Build word-index dictionaries
	- Train Word2Vec-style embeddings (negative sampling)
	- Plot training losses
	- Save embeddings + mappings
	- Word analogy / similarity checks
	- **Part 2:** NER using a Neural Network
	- **Part 2:** NER using an HMM
		- transition / emission / initial probabilities
		- Viterbi algorithm
	- Comparison section

## Methods (high level)
### Word2Vec-style training
- Preprocessing: lowercase, keep alphabetic tokens.
- Vocabulary building and word→index mapping.
- Negative sampling:
	- sampling table based on token frequency
	- extracting negative examples per center/context pair
- Optimization: gradient-based training loop in PyTorch.

### NER
- Neural model: token-level classifier (baseline) trained over features/embeddings.
- HMM model:
	- estimates transition $P(y_t \mid y_{t-1})$, emission $P(x_t \mid y_t)$, and initial $P(y_0)$
	- decodes the best tag sequence with Viterbi.

## Tech stack
- Python
- PyTorch (training / NN)
- Hugging Face Datasets (`lhoestq/conll2003`)
- NumPy, scikit-learn, Matplotlib, tqdm

## Reproducibility / How to run
1. Create and activate a virtual environment.
2. Install dependencies (example):
	 - `pip install numpy torch datasets tqdm matplotlib scikit-learn`
3. Open and run the notebook:
	 - VS Code (Jupyter extension) or Jupyter Lab/Notebook.

Notes:
- Training can take time depending on hardware.
- The notebook downloads the dataset automatically on first run.

## Project structure
```
.
├── word2vec_ner_conll2003.ipynb
└── README.md
```

## Future improvements (optional)
- Add a `requirements.txt` pinned to exact versions.
- Add a small results table/plots export (so GitHub renders results without running).
- Add a short report on error cases (confusion between PERSON/ORG/LOC, etc.).
