# -------------------------
# Build retriever + QA
# -------------------------
import os
import getpass
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import init_chat_model

# Initialize the MiniLM embedding model
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")


# llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
llm = init_chat_model("mistral-small", model_provider="mistralai")

vectorstore = Chroma(
    persist_directory="./data_store/chroma_fever_store",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------
# 7. Run a sample query
# -------------------------
query = "Was Colin Kaepernick a quarterback for the 49ers?"
result = qa_chain(query)

print("Answer:", result["result"])
print("\n--- Sources ---")
for doc in result["source_documents"]:
    print(doc.metadata, ":", doc.page_content[:200])
