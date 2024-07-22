

from  langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# loader = DirectoryLoader("docs", glob="**/*.txt")
# print("Loading documents...")
# documents = loader.load()
# print("Loaded documents: " + len(documents))

loader = TextLoader("docs/Stephen King -22 November 1963.txt")
documents = loader.load()

print ("Creating embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

print("Creating text splitter...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

print("Splitting documents...")
texts = text_splitter.split_documents(documents)

print("Creating vectorstore...")
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding= embeddings,
    persist_directory="./local-rag-db")

print("Done")