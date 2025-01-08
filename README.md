# Unified Graph and Feature-Based Embedding Framework for Cold-Start User and Item Recommendations

## Introduction
This project focuses on developing a Unified Graph and Feature-Based Embedding Framework to address the challenges of cold-start recommendations.

A report with details on data analysis and model design is available [here](https://drive.google.com/file/d/1xnApEsaXLqCnOAJr8Ho9zdmvEhTo_xPX/view?usp=sharing).

## Key Files and Folders
### 1. `run.ipynb`
- Contains the experiment result shown in the report

### 2. `movieLens_EDA.ipynb`
- Contains exploratory data analysis of the MovieLens dataset.
- Includes visualizations and insights.

### 3. `data/`
- Contains raw and preprocessed datasets.
- **Example files**:
  - `ml-100k`: MovieLens dataset.
  - `fetched_movie.csv`: dataset contains actors and directors.

### 4. `models/`
- Contains baseline models and proposed model.

### 5. `movieLens_knowledgegraph.ipynb`
- Builds a knowledge graph using movie metadata.

### 6. `train_and_evaluate.py`
- A Python script to train and evaluate recommendation models.

### 7. `utils/`
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
   cd dinghui101
