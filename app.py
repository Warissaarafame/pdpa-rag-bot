import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="PDPA Q&A Bot", page_icon="🤖")
st.title("🤖 ระบบสอบถามข้อมูล PDPA (RAG System)")

# ดึงค่า API Key จาก Secrets
api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_rag_system():
    loader = DirectoryLoader('./data', glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

try:
    if not api_key:
        st.error("กรุณาตั้งค่า GEMINI_API_KEY ใน Secrets ก่อนใช้งาน")
    else:
        retriever = load_rag_system()
        
        # ใช้ชื่อโมเดลมาตรฐาน gemini-2.0-flash พร้อมส่ง google_api_key กำกับไว้ชัดเจน
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=api_key,
            temperature=0.2
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "คุณคือผู้เชี่ยวชาญด้านกฎหมาย PDPA ตอบคำถามโดยใช้ข้อมูลจากบริบท (Context) ที่กำหนดให้เท่านั้น หากไม่พบคำตอบในบริบท ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารอ้างอิง'"),
            ("human", "Context:\n{context}\n\nคำถาม: {input}")
        ])

        rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        user_input = st.text_input("พิมพ์คำถามของคุณเกี่ยวกับ PDPA:")

        if user_input:
            with st.spinner("กำลังค้นหาคำตอบ..."):
                response = rag_chain.invoke(user_input)
                st.subheader("คำตอบ:")
                st.write(response)
                
                with st.expander("ดูเอกสารอ้างอิง"):
                    ref_docs = retriever.invoke(user_input)
                    for doc in ref_docs:
                        st.write(f"- {doc.page_content}")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดระบบ: {e}")
