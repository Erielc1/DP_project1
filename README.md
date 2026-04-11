# DP_project1
Predicting warehouse product demand using custom PyTorch neural networks and entity embeddings.


## The Business Problem
Accurate demand forecasting is critical for optimizing warehouse inventory, preventing stockouts, and reducing holding costs. This project builds a predictive model to forecast order demand for thousands of products across multiple warehouses using historical sales data.

## The Data
* **Source:** 6 years of historical product demand data (2011–2017).
* **Scope:** 2,160 unique products across 33 categories, stored in 4 different warehouses.
* **Target Variable:** `Order_Demand` (continuous variable).

## Methodology & Technical Approach
Instead of relying strictly on traditional statistical baselines, this project explores the effectiveness of **Deep Learning** for tabular supply chain data. 

* **Framework:** PyTorch
* **Categorical Handling:** Implemented **Entity Embeddings** (`nn.Embedding`) to capture latent relationships between Product Codes and Product Categories, allowing the neural network to learn complex product similarities.
* **Model Architecture:** Built and compared two custom PyTorch architectures:
  * `Model 0`: A baseline Multi-Layer Perceptron (MLP).
  * `Model 1`: A deeper network utilizing Batch Normalization and Dropout to prevent overfitting.
* **Optimization:** Used Huber Loss to handle extreme demand outliers, an Adam optimizer with weight decay, and a learning rate scheduler (`ReduceLROnPlateau`).
* **Performance:** Leveraged Mixed-Precision Training (`torch.amp.autocast`) to accelerate training on the GPU.

## Results
`Model 1` significantly outperformed the baseline network, demonstrating that batch normalization and dropout are effective at helping the model generalize unseen demand spikes. 
* **Model 0 MAE:** 1413.0
* **Model 1 MAE:** 963.6
* *Model 1 provided a more accurate forecast on 75% of the sampled test cases.*
