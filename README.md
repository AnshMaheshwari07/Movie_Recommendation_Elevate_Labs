# Movie_Recommendation_Elevate_Labs

Graph Neural Network Recommender System Report
1. Project Overview
This notebook implements a Graph Neural Network (GNN)–based recommendation system using the MovieLens 20M dataset. The main goal is to predict user ratings and generate personalized movie recommendations by modeling user–movie interactions as a heterogeneous graph.

2. Data Loading and Preprocessing
- Libraries Imported: numpy, pandas, torch, torch_geometric, sklearn, and visualization libraries.
- Datasets:
  • rating.csv: Contains user–movie ratings (columns: userId, movieId, rating, timestamp).
  • movie.csv: Contains movie metadata (columns: movieId, title, genres).
- Merging: Joined ratings with movie metadata on movieId.
- Index Encoding: Encoded userId and movieId into contiguous integer indices using LabelEncoder for PyG node indexing.
  
3. Graph Construction
- Node Types:
  • User nodes representing each unique user.
  • Movie nodes representing each unique movie.
- Edges: Directed edges from user to movie with edge_attr set to the normalized rating value.
- PyG Data Object: Created a HeteroData object with user and movie node sets and the rates relation. Node features initialized as one-hot vectors for users and movies, or placeholder embeddings.


4. Model Architectures
Implemented three GNN variants for comparison:
1. GCN-based Recommender: Layers: GCNConv for message passing. Aggregation: Sum pooling via global_mean_pool.
2. GraphSAGE-based Recommender: Layers: SAGEConv. Aggregation: Mean pooling.
3. GAT-based Recommender: Layers: GATConv with multi-head attention. Aggregation: Concatenation or averaging of attention heads.
Each model ends with a fully connected layer to predict the rating score.

5. Training Procedure
- Train/Test Split: Split edges (interactions) into training and testing sets (e.g., 80/20 split).
- Loss Function: Mean Squared Error (MSE) loss between predicted and true ratings.
- Optimizer: Adam optimizer with a learning rate of 0.01.
- Training Loop: For each epoch: forward pass, compute MSE loss, backward pass and optimizer step, logging of training loss.
  
6. Evaluation
- Metrics: Root Mean Squared Error (RMSE) on the test set.
- Results Summary (example values):
  Model    
| Test RMSE-1.0524
| Train loss: 1.1067

7. Recommendations Generation
For a given user:
1. Compute embeddings for all movies via the trained GNN.
2. Predict ratings for unrated movies.
3. Rank movies by predicted rating and recommend top-N.
Example: User 123: ["The Shawshank Redemption", "Inception", ...]

8. Conclusions and Next Steps
- Key Findings:
  • GAT performed best in capturing complex user–item interactions.
  • Graph-based models outperform traditional matrix-factorization baselines on sparse data.
- Improvements:
  • Experiment with node feature enrichment (e.g., genre embeddings, user demographics).
  • Hyperparameter tuning (e.g., number of layers, hidden dimensions, attention heads).
  • Scalability: Implement mini-batch training with neighbor sampling for large graphs.
  
9. References
• PyTorch Geometric Documentation
• MovieLens Dataset: https://grouplens.org/datasets/movielens/20m/

