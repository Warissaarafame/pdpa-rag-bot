import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="PDPA Q&A Bot", page_icon="🤖")
st.title("🤖 ระบบสอบถามข้อมูล PDPA (RAG System)")

api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GROQ_API_KEY")

@st.cache_resource
def load_rag_system(api_key):
    loader = DirectoryLoader('./data', glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # ใช้ Gemini Embedding แบบเบา ประหยัดหน่วยความจำ
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

try:
    if not api_key:
        st.error("กรุณาตั้งค่า GEMINI_API_KEY ใน Secrets ก่อนใช้งาน")
    else:
        retriever = load_rag_system(api_key)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0.2
        )

        system_prompt = (
            "คุณคือผู้เชี่ยวชาญด้านกฎหมาย PDPA ตอบคำถามโดยใช้ข้อมูลจากบริบท (Context) ที่กำหนดให้เท่านั้น "
            "หากไม่พบคำตอบในบริบท ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารอ้างอิง'\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        user_input = st.text_input("พิมพ์คำถามของคุณเกี่ยวกับ PDPA:")

        if user_input:
            with st.spinner("กำลังค้นหาคำตอบ..."):
                response = rag_chain.invoke({"input": user_input})
                st.subheader("คำตอบ:")
                st.write(response["answer"])
                
                with st.expander("ดูเอกสารอ้างอิง"):
                    for doc in response["context"]:
                        st.write(f"- {doc.page_content}")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดระบบ: {e}")
