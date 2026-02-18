import kagglehub
import shutil
import os

path = kagglehub.dataset_download("arushchillar/disneyland-reviews")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
dest_path = os.path.join(data_dir, "DisneylandReviews.csv")

shutil.copy(f"{path}/DisneylandReviews.csv", dest_path)

print("Done 👍")
