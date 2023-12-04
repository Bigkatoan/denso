
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from dotenv import load_dotenv,find_dotenv
import os
# import fitz

####################STAGE 0 LOAD CONFIG ############################
# load_dotenv(find_dotenv(),override=True)
CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
#print(CHROMADB_HOST)


#####################STAGE 1 BUILDING VECTOR DB########################
###PArt1: Document input 
def load_pdf(file_path):
 loader = PyPDFLoader(file_path= file_path)
 docs = loader.load()
 #docs = fitz.open(file_path)
 return docs


###Part2: Chunking Document
#Spliter model
def split_document(docs,chunk_size = 800, chunk_overlap = 20):
 text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ) 
    # Splitting the documents into chunks
 chunks = text_splitter.create_documents([docs])
 return chunks


###Part3: Embedding Document
#Create embedding model 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
model = HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
#model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
database = Chroma(persist_directory="./chroma_db", embedding_function=model)

#INSERT document to db
def insert_pdf_to_db(file_path):
 #Load pdf into pages
 pages = load_pdf(file_path)
 chunks = []#create empty chunks
 #insert từng chunk vào chunk
 for page in pages:
  docs = split_document(page.page_content)#Return Langchain Documents list
  
  for doc in docs:
   chunk = Document(page_content=doc.page_content, metadata=page.metadata)
   chunks.append(chunk)
##########TEsst khâu đọc
 first_page = pages[0]
 with open("result001.txt",'a',encoding='utf-8') as f:
    f.write(first_page.page_content)
###########TEst khâu qua splitter
 docs = split_document(first_page.page_content)
 for doc in docs:
   with open("result002.txt",'a',encoding='utf-8') as f:
    f.write(doc.page_content)
###############
 #Tạo DB
 db2 = Chroma.from_documents(chunks, model, persist_directory="./chroma_db")


def get_similar_chunks(query,db=database,k=5):
 chunks = db.similarity_search_with_score(query=query,k=k)
 return chunks
 
def get_response_from_query(query,chunks):
 chunks = chunks
 docs = " ".join([chunk[0].page_content for chunk in chunks if chunk[1]>33])

 from langchain.chat_models import ChatOpenAI
 from langchain.prompts import PromptTemplate
 from langchain.chains import LLMChain

 llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5,openai_api_key=OPEN_AI_API_KEY)

 prompt =PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        ###
         You are an Process assistants, you have knowledge of process, guidelines, machine document of the factory.

         Given the document bellow, Provide the instruction about the question below base on the provided provided document
         You use the tone that instructional,technical and concisely.
         Answer in Vietnamese
        ###
         Document: {docs}
         Question: {question}
        """,
    )
 chain = LLMChain(llm=llm, prompt=prompt)
 output = chain.run({'question': query, 'docs': docs})
 return output



#############TEST###############
sample_pdf_path = "sample pdf\Huong dan su dung CP1000_VN.pdf"
sample_pdf_path2 = "sample pdf/cnxh.pdf"

insert_pdf_to_db(sample_pdf_path)


query = "Tài liệu này là gì"
chunks = get_similar_chunks(query=query)
i = 1
for chunk in chunks:
 #if chunk[1]>30:
   print(i,"Score:",chunk[1])
   print("source:",chunk[0].metadata['source'],"page",chunk[0].metadata['page'])
   #print(chunk[0].page_content)
   #print(chunk[0].page_content)
   with open("result.txt",'a',encoding='utf-8') as f:
    f.write(str(i))
    #f.write("source:",chunk[0].metadata['source'],"page",chunk[0].metadata['page'])
    f.write(str(chunk[0].page_content))
    f.write("\n-------------------------------------------")
   i = i+1

#response = get_response_from_query(chunks=chunks,query=query)
#print(response)



