import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

#############################
# 1. SETUP
#############################
topics = ["politics", "science", 'sports']
vocab = ["election", "ballot", "quantum", "basketball"]
V = len(vocab)
K = len(topics)
D = 2

# Hyperparameters for Dirichlet distributions
alpha = np.array([0.5, 0.5, 0.5])  # Prior for document-topic distribution (θ) #TRY
beta = np.ones(V) * 5  # Prior for topic-word distribution (ϕ) #TRY

#############################
# 2. Generate Topic-Word Distributions (ϕ)
#############################
# Each topic's word distribution is sampled from a Dirichlet with parameter beta.
phi = np.array([np.random.dirichlet(beta) for _ in range(K)])
print("Topic-word distributions (phi):")
for i, row in enumerate(phi):
    print(f"{topics[i]}: " + " ".join(f"{x:.2f}" for x in row))

#############################
# Documents that I have (e.g. real documents downloaded from a newspaper site)
# THIS IS THE REAL TRUE HIDDEN ASSIGNMENT
#############################
documents = [
    [('election', 'politics'), ('election', 'politics'), ('ballot', 'politics'), ('ballot', 'politics'), ('quantum', 'science')],
    [('quantum', 'science'), ('quantum', 'science'), ('quantum', 'science'), ('basketball', 'sports'), ('election', 'politics')]
]

# Sample topic distribution for each document from Dirichlet(alpha)
theta = np.array([np.random.dirichlet(alpha) for _ in range(D)])

# Sample a topic for the word based on theta[d]
sampled_documents = []
for d in range(D):
    sampled_doc = []
    for word, _ in documents[d]:  # Ignore the true hidden assignment
        topic_idx = np.random.choice(K, p=theta[d])
        sampled_topic = topics[topic_idx]
        sampled_doc.append((word, sampled_topic))
    sampled_documents.append(sampled_doc)

df_sampled_documents = {
    f"Doc {d + 1}": pd.DataFrame(sampled_documents[d], columns=["Word", "Sampled Topic"])
    for d in range(D)
}
df_sampled_documents

#############################
# 4. INITIALIZE COUNT MATRICES FOR GIBBS SAMPLING
#############################
doc_topic_counts = np.zeros((D, K)) # for each document
word_topic_counts = np.zeros((K, V)) # for all documents
documents_assignments = []

for d, doc in enumerate(documents):
    doc_assignments = []
    for word, assigned_topic in doc:
        topic_idx = topics.index(assigned_topic)
        word_idx = vocab.index(word)
        doc_topic_counts[d, topic_idx] += 1
        word_topic_counts[topic_idx, word_idx] += 1
        doc_assignments.append(topic_idx)
    documents_assignments.append(doc_assignments)

pd.DataFrame(doc_topic_counts, index=["Doc 1", "Doc 2"], columns=["politics", "science", "sport"])
pd.DataFrame(word_topic_counts, index=["politics", "science", 'sports'], columns=["election", "ballot", "quantum", "basketball"])



#############################
# 5. GIBBS SAMPLING OVER ALL DOCUMENTS
#############################
num_iterations = 100  #TRY
changes_history = []

for it in range(num_iterations):
    total_changes = 0
    for d, doc in enumerate(documents):
        for i, (word, _) in enumerate(doc):
            current_topic = documents_assignments[d][i]
            word_idx = vocab.index(word)

            # Remove the current assignment (the "minus i" counts)
            doc_topic_counts[d, current_topic] -= 1
            word_topic_counts[current_topic, word_idx] -= 1

            # Compute the conditional probability for each topic
            topic_probs = np.zeros(K)
            for k in range(K):
                # Component from document-topic: how many times topic k appears in document d (plus alpha)
                doc_prob = doc_topic_counts[d, k] + alpha[k]
                # Component from topic-word: how likely word appears in topic k
                word_prob = (word_topic_counts[k, word_idx] + beta[word_idx]) / (
                            np.sum(word_topic_counts[k]) + np.sum(beta))
                topic_probs[k] = doc_prob * word_prob

            # Normalize to form a valid probability distribution
            topic_probs = topic_probs / np.sum(topic_probs)

            # Sample a new topic based on computed probabilities
            new_topic = np.random.choice(K, p=topic_probs)

            # Check if the assignment has changed
            if new_topic != current_topic:
                total_changes += 1

            # Update counts with the new assignment
            doc_topic_counts[d, new_topic] += 1
            word_topic_counts[new_topic, word_idx] += 1
            documents_assignments[d][i] = new_topic

    changes_history.append(total_changes)
    print(f"\nIteration {it + 1}: Number of topic assignment changes = {total_changes}")
    print("Document-topic counts:")
    print(doc_topic_counts)
    print("Word-topic counts:")
    print(word_topic_counts)


x = np.arange(1, num_iterations + 1)
y = np.array(changes_history)
slope, intercept = np.polyfit(x, y, 1)
reg_line = slope * x + intercept

plt.style.use("dark_background")
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='', color='cyan', label='Assignment Changes')
plt.plot(x, reg_line, 'r--', label='OLS Regression Line')
plt.xlabel("Iteration", fontsize=14, color="white")
plt.ylabel("Number of Assignment Changes", fontsize=14, color="white")
plt.title("Gibbs Sampling Convergence", fontsize=16, color="white")
plt.legend()
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
plt.savefig("plots/gibbs_sampling_convergence.png", dpi=300, bbox_inches="tight", facecolor="black")
plt.show()