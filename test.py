import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from dotenv import load_dotenv,find_dotenv
import os
import shutil
shutil.rmtree('chroma_db')
import fitz
import numpy

####################STAGE 0 LOAD CONFIG ############################
# load_dotenv(find_dotenv(),override=True)
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
#print(CHROMADB_HOST)


#####################STAGE 1 BUILDING VECTOR DB########################


###Part2: Chunking Document
#Spliter model
def split_document(docs,chunk_size = 2000, chunk_overlap = 20):
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
model = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
database = Chroma(persist_directory="./chroma_db", embedding_function=model)

#INSERT document to db
def insert_pdf_to_db(file_path):
 #Load pdf into pages
 pages = fitz.open(file_path)
 chunks = []#create empty chunks
 #insert từng chunk vào chunk
 for page in pages:
  docs = split_document(page.get_text().replace('\n', ' ').lower())#Return Langchain Documents list
  
  for doc in docs:
   chunk = Document(page_content=doc.page_content, metadata={'source': pages.name,'page': page.number})
   chunks.append(chunk)
   
 #Tạo DB
 Chroma.from_documents(chunks, model, persist_directory="./chroma_db")
 #print(chunks)

def dis(a):
   return numpy.sum(a**2)**.5

def cosine(a, b):
   A = list(a)
   B = list(b)
   while len(A) < len(B):
      A.append(-1)
   while len(B) < len(A):
      B.append(-1)
   A = numpy.array(A)
   B = numpy.array(B)
   return (A@B)/(dis(A)*dis(B))

def get_point(a, b):
   score = 0
   for t in set(a):
      if t in b:
         if b.count(t) < 3:
            score += 1/b.count(t)
         else:
            score -= 1/b.count(t)
         continue
   return score

def get_similar_chunks(query,db=database,k=10,th=1):
   chunks = db.similarity_search_with_score(query=query,k=k)
   
   query = query.lower()
   rm = ',.?!'
   for t in rm:
      query = query.replace(t, '')

   vquery, token, i = [], {}, 0
   for t in query.split(' '):
      if t not in token:
         token.update({t: i})
         vquery.append(i)
         i += 1
         continue
      vquery.append(token[t])

   al = db.get()
   docs = al['documents']
   cosines = {}
   for doc in docs:
      vdoc = []
      for t in doc.split(' '):
         if t in token:
            vdoc.append(token[t])
            continue
         vdoc.append(-1)
      if get_point(vquery, vdoc) > th:
         cosines.update({doc: get_point(vquery, vdoc)})
   return chunks, cosines, vquery
 
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
sample_pdf_path = "sample pdf/Huong dan su dung CP1000_VN.pdf"
sample_pdf_path2 = "sample pdf/LNCT800SoftwareApplicationManual-3.pdf"

insert_pdf_to_db(sample_pdf_path)
#insert_pdf_to_db(sample_pdf_path2)

insert_pdf_to_db("sample pdf/LNCT800SoftwareApplicationManual (1).pdf")
start_time = time.time()
query = "Alarm INT3170"
chunks, cosines, vquery = get_similar_chunks(query=query.lower())

for chunk in chunks:
   print("      Score:",chunk[1], end='')
   print("metadata:",chunk[0].metadata, end='')
   print(chunk[0].page_content, end='\n\n')
for i in cosines.keys():
   print(cosines[i])
   print(i, end='\n\n')

print("--- %s seconds ---" % (time.time() - start_time))
print(len(cosines.keys()))