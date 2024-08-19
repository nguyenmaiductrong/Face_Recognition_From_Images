import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

def is_match(know_embedding, candidate_embedding):
    if know_embedding.shape == candidate_embedding.shape:    
        score = cosine(know_embedding, candidate_embedding)
        return score <= 0.4, score
    return False, -1
    
def check(know_compressed_path, candidate_compressed_path):

    know_data = np.load(know_compressed_path, allow_pickle = True)
    candidate_data = np.load(candidate_compressed_path, allow_pickle = True)

    know_embeddings_dict = {**know_data}
    candidate_embeddings_dict = {**candidate_data}

    for i in candidate_embeddings_dict:
        candidate_embedding = candidate_embeddings_dict[i]
        for j in know_embeddings_dict:
            know_embedding = know_embeddings_dict[j]
            similarity, score = is_match(know_embedding, candidate_embedding)
            print(score)
            if similarity == True:
                return True
    return False