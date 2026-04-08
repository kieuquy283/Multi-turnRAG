def retrieve_context(db, query, k):
    return db.similarity_search(f"query: {query}", k=k)