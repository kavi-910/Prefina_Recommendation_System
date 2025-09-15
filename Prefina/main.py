from fastapi import FastAPI, HTTPException
from implicit.als import AlternatingLeastSquares
import numpy as np
from typing import List

# Initialize FastAPI app
app = FastAPI(title="ALS Recommendation API", description="API for serving recommendations based on ALS model", version="1.0.0")

# Placeholder for your trained model and data structures (replace with actual loaded objects)
als = AlternatingLeastSquares()  # Replace with your trained model
train_matrix = None  # Replace with your sparse matrix
idx2product = {}  # Replace with your product ID mapping

# Load your model and data (this should be done in a production setup, e.g., from a file or database)
# Example: Load from a saved model file or initialize as in your notebook
def load_model_and_data():
    global als, train_matrix, idx2product
    # Simulate loading (replace with actual loading logic)
    # als = AlternatingLeastSquares().fit(train_matrix)  # Load your trained model
    # idx2product = {...}  # Load your idx2product mapping
    pass  # Implement your loading logic here (e.g., using joblib or pickle)

# Call this at startup
load_model_and_data()

def recommend_products_numeric(user_idx: int, k: int = 5):
    """Recommend products for a given user index."""
    try:
        rec_ids, _ = als.recommend(
            userid=int(user_idx),
            user_items=train_matrix[int(user_idx)],
            N=k,
            filter_already_liked_items=True,
            recalculate_user=True
        )
        return [idx2product[i] for i in rec_ids]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in recommendation: {str(e)}")

# API Endpoints

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

@app.get("/recommend/{user_idx}", response_model=List[str])
async def get_recommendations(user_idx: int, k: int = 5):
    """
    Get top-k product recommendations for a given user index.
    - user_idx: Numeric index of the user (0 to train_matrix.shape[0]-1)
    - k: Number of recommendations (default: 5)
    """
    if user_idx < 0 or user_idx >= train_matrix.shape[0]:
        raise HTTPException(status_code=400, detail="Invalid user index")
    recommendations = recommend_products_numeric(user_idx, k)
    return recommendations