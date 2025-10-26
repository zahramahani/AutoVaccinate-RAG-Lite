from langchain_community.vectorstores import Chroma

db = Chroma(persist_directory="./data_store/chroma_fever_store")
print("✅ Collection loaded.")
print("Number of vectors:", db._collection.count())
