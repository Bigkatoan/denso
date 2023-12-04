import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def split_document(docs,chunk_size = 800, chunk_overlap = 20):
 text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ) 
    # Splitting the documents into chunks
 chunks = text_splitter.create_documents([docs])
 return chunks

file_path = "sample pdf\LNCT800SoftwareApplicationManual-3.pdf"
pages = fitz.open(file_path)

chunks = []

#doc = pages[0]
##print(doc.get_text().encode('utf-8'))
#with open("debug.txt",'a',encoding='utf-8') as f:
#    f.write(doc.get_text())

for page in pages:
    docs = split_document(page.get_text().replace('\n', ' '))
    for doc in docs:
       chunk = Document(page_content=doc.page_content,metadata={'source': pages.name,'page': page.number})
       chunks.append(chunk)

#print(doc.name)
print((chunks[0]))