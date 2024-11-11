# 기본 데이터 처리 및 유틸리티 라이브러리
import os
import random
import numpy as np
import pandas as pd
import argparse

# PyTorch 및 Hugging Face 관련 라이브러리
import torch
from transformers import AutoTokenizer, AutoModel

# 시각화 라이브러리
import matplotlib.pyplot as plt

# 머신러닝 및 통계 관련 라이브러리
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_pooling(model_output, attention_mask):
    """
    Generates a sentence embedding by mean pooling the token embeddings from the model output, 
    using the attention mask to ignore padding tokens.

    Parameters:
    - model_output (torch.Tensor): The last hidden state of the model output (from transformers like BERT).
        - shape: (batch_size, sequence_length, hidden_size)
    - attention_mask (torch.Tensor): A mask indicating the valid token positions in the input sentence.
        - shape: (batch_size, sequence_length)

    Returns:
    - torch.Tensor: The mean-pooled sentence embedding.
        - shape: (batch_size, hidden_size)
    """
    token_embeddings = model_output.last_hidden_state  # Embedding of all tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embedding(texts, model_name, saved_model_path = None) :
    """
    Generates embeddings for the input text data using a specified model.

    Parameters:
    - texts (List[str]): A list of text data for which to generate embeddings.
    - model_name (str): The name of the Hugging Face model to use for embedding generation.
    - saved_model_path (str, optional): Path to a saved model. 
        - If specified, the model is loaded from this path instead of from `model_name`.
    
    Returns:
    - embeddings (numpy.ndarray): An array of sentence embeddings for each input text.
        - Shape: (num_texts, hidden_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if saved_model_path is not None :
        model = AutoModel.from_pretrained(saved_model_path).to(DEVICE)
    else :
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
    
    embeddings = []
    model.eval()
    for text in texts :
        with torch.no_grad() :
            input = tokenizer(text, return_tensors = 'pt').to(DEVICE)
            model_output = model(**input)
            embeddings.append(mean_pooling(model_output, input['attention_mask']))

    embeddings = torch.concat(embeddings, dim = 0)
    embeddings = embeddings.cpu().numpy()

    return embeddings


def save_plot(x_values, y_values, targets, preds, save_path, lim_range = 0.5) :
    """
    Generates and saves scatter plots comparing actual target values and predicted values.

    Parameters:
    - x_values (list or array-like): The x-axis values for the scatter plots.
    - y_values (list or array-like): The y-axis values for the scatter plots.
    - targets (list or array-like): Actual target values used for color coding the first scatter plot.
    - preds (list or array-like): Predicted values used for color coding the second scatter plot.
    - save_path (str): Path where the generated plot image will be saved.
    - lim_range (float, optional): Scaling factor to adjust the plot's x and y axis limits. Default is 0.5.

    Returns:
    - None: The function saves the plot as an image and does not return any values.
    """
    fig, axes = plt.subplots(1,2, figsize = (16,6))

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    targets_scatter = axes[0].scatter(x_values, y_values,
                                      c = targets, cmap = 'viridis', alpha = 0.7)
    axes[0].set_title("Actual Targets")
    fig.colorbar(targets_scatter, ax = axes[0])
    axes[0].set_xlim(x_min * lim_range, x_max * lim_range)
    axes[0].set_ylim(y_min * lim_range, y_max * lim_range)

    preds_scatter = axes[1].scatter(x_values, y_values,
                                    c = preds, cmap = 'virids', alpha = 0.7)
    axes[1].set_title("Predicted by Clusters")
    fig.colorbar(preds_scatter, ax = axes[1])
    axes[1].set_xlim(x_min * lim_range, x_max * lim_range)
    axes[1].set_ylim(y_min * lim_range, y_max * lim_range)

    plt.savefig(save_path, format = 'png')
    plt.close()


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help="Path to the input data CSV file.")
    parser.add_argument("--model_name", type=str,
                        help="Name of the Hugging Face model to use for embedding generation.")
    parser.add_argument("--saved_model_path", type=str, default=None,
                        help="Path to a pre-trained model for embedding generation. If not provided, the model specified by model_name will be used.")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save the output files, including new label mappings and clustering results.")
    parser.add_argument("--use_all", type=bool, default=False,
                        help="If set to True, uses the entire dataset for clustering. If False, uses only the first 1600 samples.")
    
    args = parser.parse_args()

    os.mkdir(args.output_dir, exist_ok =True)
    
    data = pd.read_csv(args.data_path)
    origin_data = data.copy()
    if not args.use_all :
        data = data.iloc[:1600, :]
    
    labels = data['target'].values
    texts = data['text'].values

    embeddings = get_embedding(texts, args.model_name, args.saved_model_path)

    cluster = KMeans(n_clusters=len(labels.unique()), random_state=SEED)
    predicted_label = cluster.fit_predict(embeddings)

    # Applying the Hungarian Algorithm 
    conf_matrix = confusion_matrix(data['target'], predicted_label)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    mappings = {col : row for row, col in zip(row_ind, col_ind)}
    centriods = cluster.cluster_centers_

    if args.use_all : 
        rest_embeddings = get_embedding(texts[1600:], args.model_name, args.saved_model_path)
        cluster_labels = torch.argmax(torch.tensor(rest_embeddings @ centriods.T), dim = -1).numpy()
        mapping_labels = [mappings[c_label] for c_label in cluster_labels]

        predicted_label.extend(mapping_labels)

    origin_data['target'] = predicted_label
    origin_data.to_csv(os.path.join(args.output_dir, "new_mapping_label.csv"))

    # Performimg PCA for visualization.
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance_ratio = pca.explained_variance_ratio_

    with open(os.path.join(args.output_dir, "cluster_results.txt"), "w") as f: 
        f.write(f"Confuse matrix :\n{conf_matrix}")
        f.write(f"Mappint results :\n{mappings}")
        f.write(f"Percentage of variance explained : {explained_variance_ratio}")
        f.write(f"Cumulative Percentage of Variance Explained : {explained_variance_ratio.cumsum()}")

    save_plot(reduced_embeddings[:,0], reduced_embeddings[:,1], labels, predicted_label,
              os.path.join(args.output_dir, "cluster_img.png"))


if __name__ == "__main__":
    main()