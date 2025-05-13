# Expedia Hotel Recommendation System

This project implements a hotel recommendation system for Expedia, predicting which hotels users are most likely to book based on their search queries and historical data.

## Project Structure

- `src/`: Source code directory
  - `data/`: Data handling modules
  - `models/`: Model implementation
  - `utils/`: Utility functions
  - `visualization/`: Visualization modules
- `notebooks/`: Jupyter notebooks for analysis and experimentation
- `docs/`: Documentation files
- `tests/`: Unit tests

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

The dataset can be downloaded from the Kaggle competition:
https://www.kaggle.com/competitions/dmt-2025-2nd-assignment

## Project Tasks

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling and Evaluation
5. Deployment

## Evaluation Metric

The model is evaluated using Normalized Discounted Cumulative Gain (NDCG)@5, calculated per query and averaged over all queries with values weighted by the log₂ function.

## Team

[Your team information here]

## License

[Your license information here]
