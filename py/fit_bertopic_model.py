import pandas as pd
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from llm import build_llm_representation


# --- Load Data --- #
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
data_path = os.path.join(workspace_dir, "data", "DisneyReviews_processed.csv")
df = pd.read_csv(data_path)

print("Data loaded successfully!")


# --- Group Data by Branch and Sentiment --- #
branches = df["Branch"].unique()
branch_dfs = {}

for branch in branches:
    subset = df[(df["Branch"] == branch)]
    branch_dfs[branch] = subset
    print(f"Branch: {branch}, Number of Reviews: {len(subset)}")

print("Data grouped successfully!")

# --- Global Models --- #
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
base_umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
base_hdbscan = HDBSCAN(min_cluster_size=25, min_samples=2)
llm_representation = build_llm_representation()
print("Global model set!")

# Create output directory
model_output_dir = os.path.join(workspace_dir, "models/topic_models")
os.makedirs(model_output_dir, exist_ok=True)
print("Directory created!")

for branch, branch_df in branch_dfs.items():
    print(f"\nFitting model for {branch}...")

    reviews = branch_df["review_clean"].dropna().astype(str).tolist()

    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=base_umap,
        hdbscan_model=base_hdbscan,
        representation_model={"LLM": llm_representation}, 
        verbose=True
    )

    topics, probabilities = topic_model.fit_transform(reviews)
    print(topic_model.get_topic_info().head())
