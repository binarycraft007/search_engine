import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    def __init__(self):
        # Dummy corpus for demonstration
        self.documents = [
            {
                "id": 1,
                "title": "Introduction to Information Retrieval",
                "content": "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources.",
            },
            {
                "id": 2,
                "title": "Python Programming",
                "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            },
            {
                "id": 3,
                "title": "Flask Web Framework",
                "content": "Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries.",
            },
            {
                "id": 4,
                "title": "Machine Learning Basics",
                "content": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
            },
            {
                "id": 5,
                "title": "Search Engine Optimization",
                "content": "Search engine optimization is the process of improving the quality and quantity of website traffic to a website or a web page from search engines.",
            },
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self._index_documents()

    def _index_documents(self):
        """Indexes the documents using TF-IDF."""
        corpus = [doc["content"] for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=5):
        """
        Searches the corpus for the query.
        Returns a list of relevant documents with their scores.
        """
        if not query:
            return []

        # Transform the query to the same vector space
        query_vec = self.vectorizer.transform([query])

        # Calculate cosine similarity between query and all documents
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get indices of documents sorted by similarity (descending)
        related_docs_indices = cosine_similarities.argsort()[::-1]

        results = []
        for i in related_docs_indices:
            score = cosine_similarities[i]
            if score > 0:  # Only return relevant results
                doc = self.documents[i].copy()
                doc["score"] = round(float(score), 4)
                results.append(doc)

            if len(results) >= top_k:
                break

        return results
