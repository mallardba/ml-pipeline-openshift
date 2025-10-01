# Embedding Models, Vector Databases, and Reranking: Complete Guide

## Table of Contents
1. [Embedding Models](#embedding-models)
2. [Vector Databases](#vector-databases)
3. [Reranking Models](#reranking-models)
4. [Complete RAG Pipeline](#complete-rag-pipeline)

---

## Embedding Models

### What are Embeddings?

Embeddings are dense vector representations that capture semantic meaning of text, images, or other data types. They transform discrete tokens into continuous vectors in high-dimensional space, where similar items are close together.

### Types of Embedding Models

#### 1. Word Embeddings

**Word2Vec**: Learns embeddings by predicting context words
- **CBOW**: Predicts target word from context
- **Skip-gram**: Predicts context from target word

```python
import numpy as np
from gensim.models import Word2Vec

# Train Word2Vec model
sentences = [
    ['machine', 'learning', 'is', 'powerful'],
    ['deep', 'learning', 'uses', 'neural', 'networks'],
    ['natural', 'language', 'processing', 'is', 'fascinating']
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word embeddings
word_vector = model.wv['machine']
print(f"Machine embedding shape: {word_vector.shape}")

# Find similar words
similar_words = model.wv.most_similar('learning', topn=3)
print(f"Words similar to 'learning': {similar_words}")
```

**GloVe**: Global Vectors for Word Representation
- Combines global statistics with local context
- Uses co-occurrence matrix factorization

```python
from glove import Glove

# Train GloVe model
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(sentences, epochs=10)

# Get embeddings
word_vector = glove.word_vectors[glove.dictionary['machine']]
```

#### 2. Sentence Embeddings

**Sentence-BERT**: BERT-based sentence embeddings
- Uses siamese network architecture
- Trained on sentence pairs for semantic similarity

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing handles text"
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")

# Compute similarity
similarity_matrix = np.dot(embeddings, embeddings.T)
print(f"Similarity matrix:\n{similarity_matrix}")
```

**Universal Sentence Encoder**: Google's multilingual sentence encoder

```python
import tensorflow_hub as hub

# Load USE model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Encode sentences
embeddings = use_model(sentences)
print(f"USE embedding shape: {embeddings.shape}")
```

#### 3. Custom Embedding Models

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CustomEmbeddingModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = nn.Linear(self.model.config.hidden_size, 768)
    
    def forward(self, texts):
        # Tokenize
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token for sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project to desired dimension
        embeddings = self.pooling(embeddings)
        
        return embeddings

# Example usage
embedding_model = CustomEmbeddingModel()
texts = ["Hello world", "Machine learning is great"]
embeddings = embedding_model(texts)
print(f"Custom embeddings shape: {embeddings.shape}")
```

### Embedding Training Strategies

#### 1. Contrastive Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings1, embeddings2, labels):
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Contrastive loss
        loss = F.cross_entropy(similarity, labels)
        return loss

# Example training loop
def train_contrastive(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ContrastiveLoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            texts1, texts2, labels = batch
            
            embeddings1 = model(texts1)
            embeddings2 = model(texts2)
            
            loss = criterion(embeddings1, embeddings2, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

#### 2. Triplet Loss

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def train_triplet(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = TripletLoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            anchor, positive, negative = batch
            
            anchor_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## Vector Databases

### What are Vector Databases?

Vector databases are specialized databases designed to store, index, and search high-dimensional vectors efficiently. They enable fast similarity search and are essential for RAG systems.

### Vector Database Internals

#### 1. Indexing Algorithms

**Flat Index**: Brute force search
- **Pros**: Exact results, simple implementation
- **Cons**: Slow for large datasets

```python
import numpy as np

class FlatIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []
        self.ids = []
    
    def add(self, vector, id):
        self.vectors.append(vector)
        self.ids.append(id)
    
    def search(self, query_vector, k=10):
        # Compute distances to all vectors
        distances = []
        for vector in self.vectors:
            dist = np.linalg.norm(query_vector - vector)
            distances.append(dist)
        
        # Get top-k results
        indices = np.argsort(distances)[:k]
        return [(self.ids[i], distances[i]) for i in indices]

# Example usage
index = FlatIndex(128)
for i in range(1000):
    vector = np.random.randn(128)
    index.add(vector, i)

query = np.random.randn(128)
results = index.search(query, k=5)
print(f"Top 5 results: {results}")
```

**IVF (Inverted File) Index**: Cluster-based indexing
- Divides vectors into clusters
- Searches only relevant clusters

```python
from sklearn.cluster import KMeans

class IVFIndex:
    def __init__(self, dimension, n_clusters=100):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.clusters = None
        self.vectors = []
        self.ids = []
        self.cluster_assignments = []
    
    def train(self, vectors):
        """Train clustering model"""
        self.clusters = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters.fit(vectors)
    
    def add(self, vector, id):
        self.vectors.append(vector)
        self.ids.append(id)
        
        # Assign to cluster
        cluster_id = self.clusters.predict([vector])[0]
        self.cluster_assignments.append(cluster_id)
    
    def search(self, query_vector, k=10, n_probe=10):
        # Find closest clusters
        cluster_distances = self.clusters.transform([query_vector])[0]
        closest_clusters = np.argsort(cluster_distances)[:n_probe]
        
        # Search only in closest clusters
        candidate_indices = []
        for cluster_id in closest_clusters:
            cluster_indices = [i for i, c in enumerate(self.cluster_assignments) if c == cluster_id]
            candidate_indices.extend(cluster_indices)
        
        # Compute distances for candidates
        distances = []
        for idx in candidate_indices:
            dist = np.linalg.norm(query_vector - self.vectors[idx])
            distances.append((self.ids[idx], dist))
        
        # Sort and return top-k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

# Example usage
ivf_index = IVFIndex(128, n_clusters=50)

# Train on sample data
sample_vectors = np.random.randn(1000, 128)
ivf_index.train(sample_vectors)

# Add vectors
for i in range(1000):
    vector = np.random.randn(128)
    ivf_index.add(vector, i)

query = np.random.randn(128)
results = ivf_index.search(query, k=5)
print(f"IVF results: {results}")
```

**HNSW (Hierarchical Navigable Small World)**: Graph-based indexing
- Builds hierarchical graph structure
- Very fast approximate search

```python
import heapq
from collections import defaultdict

class HNSWIndex:
    def __init__(self, dimension, m=16, ef_construction=200):
        self.dimension = dimension
        self.m = m  # Maximum connections per node
        self.ef_construction = ef_construction
        self.graphs = []  # Multiple layers
        self.vectors = []
        self.ids = []
    
    def _distance(self, a, b):
        return np.linalg.norm(a - b)
    
    def _search_layer(self, query, candidates, ef, layer):
        """Search in a specific layer"""
        visited = set()
        candidates_heap = []
        
        for candidate in candidates:
            heapq.heappush(candidates_heap, (self._distance(query, self.vectors[candidate]), candidate))
            visited.add(candidate)
        
        while candidates_heap:
            dist, current = heapq.heappop(candidates_heap)
            
            # Check neighbors
            for neighbor in self.graphs[layer].get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_dist = self._distance(query, self.vectors[neighbor])
                    
                    if len(candidates_heap) < ef or neighbor_dist < candidates_heap[0][0]:
                        heapq.heappush(candidates_heap, (neighbor_dist, neighbor))
        
        return [candidate for _, candidate in candidates_heap]
    
    def add(self, vector, id):
        """Add vector to index"""
        self.vectors.append(vector)
        self.ids.append(id)
        vector_id = len(self.vectors) - 1
        
        # Determine layer (higher layers have fewer nodes)
        layer = 0
        while np.random.random() < 0.5 and layer < len(self.graphs):
            layer += 1
        
        # Extend graphs if necessary
        while len(self.graphs) <= layer:
            self.graphs.append(defaultdict(list))
        
        # Connect to existing nodes
        if layer == 0:
            # Bottom layer - connect to all previous nodes
            for i in range(vector_id):
                if len(self.graphs[0][i]) < self.m:
                    self.graphs[0][i].append(vector_id)
                    self.graphs[0][vector_id].append(i)
        else:
            # Higher layers - selective connections
            candidates = list(range(vector_id))
            for l in range(layer):
                candidates = self._search_layer(vector, candidates, self.ef_construction, l)
            
            # Connect to closest nodes
            distances = [(self._distance(vector, self.vectors[c]), c) for c in candidates]
            distances.sort()
            
            for _, candidate in distances[:self.m]:
                self.graphs[layer][candidate].append(vector_id)
                self.graphs[layer][vector_id].append(candidate)
    
    def search(self, query_vector, k=10, ef=50):
        """Search for similar vectors"""
        if not self.vectors:
            return []
        
        # Start from top layer
        current = 0
        for layer in range(len(self.graphs) - 1, 0, -1):
            current = self._search_layer(query_vector, [current], ef, layer)[0]
        
        # Search in bottom layer
        candidates = self._search_layer(query_vector, [current], ef, 0)
        
        # Return top-k results
        distances = [(self._distance(query_vector, self.vectors[c]), self.ids[c]) for c in candidates]
        distances.sort()
        return distances[:k]

# Example usage
hnsw_index = HNSWIndex(128)

# Add vectors
for i in range(1000):
    vector = np.random.randn(128)
    hnsw_index.add(vector, i)

query = np.random.randn(128)
results = hnsw_index.search(query, k=5)
print(f"HNSW results: {results}")
```

#### 2. Quantization Techniques

**Product Quantization**: Compress vectors for faster search

```python
class ProductQuantizer:
    def __init__(self, dimension, n_subvectors=8, n_clusters=256):
        self.dimension = dimension
        self.n_subvectors = n_subvectors
        self.subvector_dim = dimension // n_subvectors
        self.n_clusters = n_clusters
        self.codebooks = []
        self.trained = False
    
    def train(self, vectors):
        """Train quantizer on sample vectors"""
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            # Cluster subvectors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(subvectors)
            
            self.codebooks.append(kmeans)
        
        self.trained = True
    
    def encode(self, vector):
        """Encode vector to quantized representation"""
        if not self.trained:
            raise ValueError("Quantizer not trained")
        
        codes = []
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            subvector = vector[start_idx:end_idx]
            
            code = self.codebooks[i].predict([subvector])[0]
            codes.append(code)
        
        return codes
    
    def decode(self, codes):
        """Decode quantized representation back to vector"""
        if not self.trained:
            raise ValueError("Quantizer not trained")
        
        vector = np.zeros(self.dimension)
        for i, code in enumerate(codes):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            
            centroid = self.codebooks[i].cluster_centers_[code]
            vector[start_idx:end_idx] = centroid
        
        return vector

# Example usage
pq = ProductQuantizer(128, n_subvectors=8, n_clusters=256)

# Train on sample data
sample_vectors = np.random.randn(1000, 128)
pq.train(sample_vectors)

# Encode and decode
vector = np.random.randn(128)
codes = pq.encode(vector)
reconstructed = pq.decode(codes)

print(f"Original vector shape: {vector.shape}")
print(f"Codes: {codes}")
print(f"Reconstruction error: {np.linalg.norm(vector - reconstructed):.4f}")
```

#### 3. Popular Vector Databases

**Pinecone**: Managed vector database
```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")
index = pinecone.Index("your-index-name")

# Upsert vectors
vectors = [
    ("1", [0.1, 0.2, 0.3], {"text": "Machine learning"}),
    ("2", [0.4, 0.5, 0.6], {"text": "Deep learning"})
]
index.upsert(vectors)

# Query
query_vector = [0.1, 0.2, 0.3]
results = index.query(query_vector, top_k=5, include_metadata=True)
print(f"Pinecone results: {results}")
```

**Weaviate**: Open-source vector database
```python
import weaviate

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Create schema
schema = {
    "class": "Document",
    "properties": [
        {"name": "text", "dataType": ["text"]},
        {"name": "vector", "dataType": ["number[]"]}
    ]
}
client.schema.create_class(schema)

# Add objects
client.data_object.create({
    "text": "Machine learning is powerful",
    "vector": [0.1, 0.2, 0.3]
}, "Document")

# Query
query = {
    "concepts": ["machine learning"],
    "limit": 5
}
results = client.query.get("Document", ["text"]).with_near_text(query).do()
print(f"Weaviate results: {results}")
```

**Chroma**: Lightweight vector database
```python
import chromadb

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["Machine learning is powerful", "Deep learning uses neural networks"],
    ids=["1", "2"],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
)

# Query
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=2
)
print(f"Chroma results: {results}")
```

---

## Reranking Models

### What is Reranking?

Reranking is the process of reordering search results using more sophisticated models than the initial retrieval. It improves precision by using cross-attention between query and documents.

### Types of Reranking Models

#### 1. Cross-Encoder Reranking

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CrossEncoderReranker(nn.Module):
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, queries, documents):
        """Compute relevance scores for query-document pairs"""
        scores = []
        
        for query, doc in zip(queries, documents):
            # Concatenate query and document
            text = f"{query} [SEP] {doc}"
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  max_length=512, truncation=True, padding=True)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token for classification
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                score = self.classifier(cls_embedding)
                scores.append(score.item())
        
        return scores

# Example usage
reranker = CrossEncoderReranker()

queries = ["What is machine learning?"]
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing handles text"
]

scores = reranker(queries * len(documents), documents)
print(f"Reranking scores: {scores}")

# Sort documents by score
sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
print("Reranked documents:")
for doc, score in sorted_docs:
    print(f"Score: {score:.3f} - {doc}")
```

#### 2. Bi-Encoder Reranking

```python
class BiEncoderReranker(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.query_projection = nn.Linear(self.model.config.hidden_size, 256)
        self.doc_projection = nn.Linear(self.model.config.hidden_size, 256)
    
    def encode_queries(self, queries):
        """Encode queries"""
        embeddings = []
        for query in queries:
            inputs = self.tokenizer(query, return_tensors='pt', 
                                  max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embedding = self.query_projection(embedding)
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)
    
    def encode_documents(self, documents):
        """Encode documents"""
        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors='pt', 
                                  max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embedding = self.doc_projection(embedding)
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)
    
    def compute_scores(self, query_embeddings, doc_embeddings):
        """Compute similarity scores"""
        # Normalize embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=1)
        
        # Compute cosine similarity
        scores = torch.matmul(query_embeddings, doc_embeddings.T)
        return scores

# Example usage
bi_reranker = BiEncoderReranker()

queries = ["What is machine learning?"]
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing handles text"
]

query_embeddings = bi_reranker.encode_queries(queries)
doc_embeddings = bi_reranker.encode_documents(documents)
scores = bi_reranker.compute_scores(query_embeddings, doc_embeddings)

print(f"Bi-encoder scores: {scores[0].tolist()}")
```

#### 3. Multi-Stage Reranking

```python
class MultiStageReranker:
    def __init__(self):
        self.stage1_model = None  # Fast, approximate model
        self.stage2_model = None  # Slow, accurate model
    
    def rerank(self, query, documents, top_k_stage1=100, top_k_final=10):
        """Multi-stage reranking pipeline"""
        
        # Stage 1: Fast reranking (e.g., bi-encoder)
        stage1_scores = self.stage1_rerank(query, documents)
        stage1_ranked = sorted(zip(documents, stage1_scores), 
                              key=lambda x: x[1], reverse=True)
        
        # Take top-k from stage 1
        top_docs = [doc for doc, _ in stage1_ranked[:top_k_stage1]]
        
        # Stage 2: Accurate reranking (e.g., cross-encoder)
        stage2_scores = self.stage2_rerank(query, top_docs)
        stage2_ranked = sorted(zip(top_docs, stage2_scores), 
                              key=lambda x: x[1], reverse=True)
        
        # Return final top-k
        return stage2_ranked[:top_k_final]
    
    def stage1_rerank(self, query, documents):
        """Fast reranking stage"""
        # Implement fast reranking (e.g., using pre-computed embeddings)
        pass
    
    def stage2_rerank(self, query, documents):
        """Accurate reranking stage"""
        # Implement accurate reranking (e.g., cross-encoder)
        pass
```

---

## Complete RAG Pipeline

### End-to-End RAG System

```python
class CompleteRAGSystem:
    def __init__(self, embedding_model, vector_db, reranker, llm):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.reranker = reranker
        self.llm = llm
    
    def add_documents(self, documents):
        """Add documents to knowledge base"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Add to vector database
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            self.vector_db.add(emb, f"doc_{i}", metadata={"text": doc})
    
    def retrieve_and_rerank(self, query, top_k_retrieve=50, top_k_rerank=5):
        """Retrieve and rerank documents"""
        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Step 2: Retrieve from vector database
        retrieved_results = self.vector_db.search(query_embedding, k=top_k_retrieve)
        retrieved_docs = [result['metadata']['text'] for result in retrieved_results]
        
        # Step 3: Rerank documents
        rerank_scores = self.reranker([query] * len(retrieved_docs), retrieved_docs)
        
        # Step 4: Sort by rerank scores
        reranked_docs = sorted(zip(retrieved_docs, rerank_scores), 
                              key=lambda x: x[1], reverse=True)
        
        return reranked_docs[:top_k_rerank]
    
    def generate_response(self, query, max_length=200):
        """Generate response using RAG"""
        # Retrieve and rerank relevant documents
        relevant_docs = self.retrieve_and_rerank(query)
        
        # Create context
        context = "\n".join([f"- {doc}" for doc, _ in relevant_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.llm.generate(prompt, max_length=max_length)
        
        return {
            'response': response,
            'relevant_docs': relevant_docs,
            'context': context
        }

# Example usage
rag_system = CompleteRAGSystem(embedding_model, vector_db, reranker, llm)

# Add documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing deals with text and speech",
    "Computer vision focuses on image and video analysis"
]
rag_system.add_documents(documents)

# Query
query = "What is machine learning?"
result = rag_system.generate_response(query)
print(f"Response: {result['response']}")
print(f"Relevant documents: {result['relevant_docs']}")
```

### Performance Optimization

#### 1. Caching Strategies

```python
import redis
import pickle

class CachedRAGSystem(CompleteRAGSystem):
    def __init__(self, *args, cache_host='localhost', cache_port=6379):
        super().__init__(*args)
        self.cache = redis.Redis(host=cache_host, port=cache_port)
        self.cache_ttl = 3600  # 1 hour
    
    def _get_cache_key(self, query):
        """Generate cache key for query"""
        return f"rag_cache:{hash(query)}"
    
    def generate_response(self, query, max_length=200):
        """Generate response with caching"""
        cache_key = self._get_cache_key(query)
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return pickle.loads(cached_result)
        
        # Generate response
        result = super().generate_response(query, max_length)
        
        # Cache result
        self.cache.setex(cache_key, self.cache_ttl, pickle.dumps(result))
        
        return result
```

#### 2. Batch Processing

```python
class BatchRAGSystem(CompleteRAGSystem):
    def batch_retrieve_and_rerank(self, queries, top_k_retrieve=50, top_k_rerank=5):
        """Batch process multiple queries"""
        # Generate query embeddings
        query_embeddings = self.embedding_model.encode(queries)
        
        # Batch retrieve
        all_results = []
        for query_emb in query_embeddings:
            results = self.vector_db.search(query_emb, k=top_k_retrieve)
            all_results.append(results)
        
        # Batch rerank
        batch_queries = []
        batch_docs = []
        for i, results in enumerate(all_results):
            docs = [result['metadata']['text'] for result in results]
            batch_queries.extend([queries[i]] * len(docs))
            batch_docs.extend(docs)
        
        batch_scores = self.reranker(batch_queries, batch_docs)
        
        # Organize results
        final_results = []
        start_idx = 0
        for results in all_results:
            end_idx = start_idx + len(results)
            scores = batch_scores[start_idx:end_idx]
            
            reranked = sorted(zip([r['metadata']['text'] for r in results], scores),
                            key=lambda x: x[1], reverse=True)
            final_results.append(reranked[:top_k_rerank])
            start_idx = end_idx
        
        return final_results
```

This comprehensive guide covers embedding models, vector databases, and reranking systems in detail, including their internals, implementations, and how they work together in complete RAG pipelines.
