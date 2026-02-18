import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --- Load Data --- #
df = pd.read_csv('../data/DisneyReviews_processed.csv')

print("Data loaded successfully.")