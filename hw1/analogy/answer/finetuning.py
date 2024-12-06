"""

finetuning the pre-trained word vectors with the dev.txt quarternion

General steps
1. Build our loss based on the first 3 words - the last word(predicted)
2. Update our word vectors based on the loss
3. Save our finetuned model

"""

import numpy as np
from gensim.models import KeyedVectors
import torch
import torch.optim as optim
import gensim.downloader as api

# Load the pre-trained word vectors (GloVe)
word_vectors = api.load("glove-wiki-gigaword-100")

# Example of loading your analogy training data
def load_analogy_data(filename):
    analogy_data = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip the header and split into words
            if line.strip():
                words = line.strip().split()
                if len(words) == 4:  # Make sure each line has 4 words
                    analogy_data.append(words)
    return analogy_data

# Load your analogy dataset (with 4 words per line)
train_data = load_analogy_data('../data/train/dev_combined.txt')

# Define analogy loss
def analogy_loss(pytorch_word_vectors, a, a_star, b, b_star):
    # Get the word vectors
    v_a = pytorch_word_vectors[a]
    v_a_star = pytorch_word_vectors[a_star]
    v_b = pytorch_word_vectors[b]
    v_b_star = pytorch_word_vectors[b_star]  # This is what we want to predict
    
    # Compute the predicted vector for b_star using the analogy equation
    predicted_b_star = v_b + (v_a_star - v_a)
    
    # Return L2 loss between predicted and actual b_star vector
    loss = torch.norm(predicted_b_star - v_b_star)
    return loss


# Preallocate PyTorch tensors for all word vectors
pytorch_word_vectors = {word: torch.tensor(word_vectors[word], requires_grad=True) for word in word_vectors.key_to_index.keys()}

# Prepare optimizer for updating only the involved word vectors
optimizer = optim.Adam([pytorch_word_vectors[word] for word in pytorch_word_vectors.keys()], lr=0.01)

# Training
for epoch in range(10):  # Number of epochs
    total_loss = 0
    
    for analogy in train_data:
        a, a_star, b, b_star = analogy
        
        # Check if all words are in the vocabulary
        if all(word in pytorch_word_vectors for word in analogy):
            optimizer.zero_grad()
            
            # Compute loss for the current analogy
            loss = analogy_loss(pytorch_word_vectors, a, a_star, b, b_star)
            total_loss += loss.item()
            
            # Backpropagate and update word vectors
            loss.backward()
            optimizer.step()
    
    print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

# Save the finetuned vectors
def save_word_vectors(word_vecs, pytorch_word_vectors, filename):
    # Get the number of words and the dimensionality of the vectors as the Gensim is expecting
    num_words = len(word_vecs.index_to_key)
    vector_size = word_vecs.vector_size
    
    # Open the file and write the word vectors
    with open(filename, 'w') as f:
        # Write the header with number of words and vector size as Gensim is expecting
        f.write(f"{num_words} {vector_size}\n")
        
        # Iterate over all words in the vocabulary
        for word in word_vecs.index_to_key:  # Use index_to_key to get the words
            vector = pytorch_word_vectors[word].detach().numpy()  # Convert PyTorch tensor back to NumPy
            vector_str = ' '.join(map(str, vector))  # Convert vector to space-separated string
            f.write(f"{word} {vector_str}\n")  # Write the word and its vector

# Save the updated word vectors to the current working directory
save_word_vectors(word_vectors, pytorch_word_vectors, 'finetuned_glove.txt')
print("Updated word vectors have been saved to 'finetuned_glove.txt'")
