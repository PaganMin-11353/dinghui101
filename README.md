## Introduction
Unified Graph and Feature-Based Embedding Framework for Cold-Start User and Item Recommendations


## Key Files and Folders

### 1. `data/`
- Contains raw and preprocessed datasets.
- **Example files**:
  - `ml-100k`: MovieLens dataset.
  - `fetched_movie.csv`: dataset contains actors and directors.

### 2. `models/`
- Contains baseline models and proposed model.

### 3. `movieLens_EDA.ipynb`
- A Jupyter Notebook containing exploratory data analysis of the MovieLens dataset.
- Includes visualizations and insights.

### 4. `movieLens_knowledgegraph.ipynb`
- Builds a knowledge graph using movie metadata.

### 5. `train_and_evaluate.py`
- A Python script to train and evaluate recommendation models.

### 6. `utils/`
- Contains helper functions for:
  - Data preprocessing.
  - Metrics calculation.

## Getting Started

### Prerequisites
- Python >= 3.8
- Conda (optional but recommended)
- Jupyter Notebook

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/project-name.git
   cd project-name
