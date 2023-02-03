from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine



engine = create_engine("sqlite:///test.db", echo=True)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

