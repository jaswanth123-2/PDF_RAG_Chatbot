import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
import gradio as gr

def load_docs():
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

# Local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore_path = "vectorstore"

if not os.path.exists(vectorstore_path):
    print("Creating new vectorstore from PDFs...")
    texts = load_docs()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vectorstore_path)
else:
    print("Loading existing vectorstore...")
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Local LLM running on GPU
# First run: ollama pull llama3.2:3b
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
    num_predict=512
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

def chat(message, history):
    if not message.strip():
        return history
    
    result = qa_chain.invoke({"query": message})
    answer = result["result"]
    
    sources = set()
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "Unknown")
        sources.add(os.path.basename(source))
    
    source_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sorted(sources)) if sources else ""
    full_response = answer + source_text
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})
    
    return history

# Sample questions based on the actual PDFs
sample_questions = [
    "What are the levels of AGI as defined in the paper?",
    "How do reasoning models differ from standard language models?",
    "What expert-level scientific tasks can AI perform according to the evaluation?",
    "What are the key findings about AI adoption across different geographic regions?",
    "What fragility issues exist in mathematical reasoning for large language models?",
    "What do AI researchers predict about the future of artificial intelligence?",
    "What evidence suggests sparks of artificial general intelligence in GPT-4?"
]

# Gradio UI
with gr.Blocks(title="Research Papers Q&A System") as demo:
    gr.Markdown("# Research Papers Question Answering System")
    gr.Markdown("Ask questions about the AI research papers in the knowledge base.")
    
    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        placeholder="Your questions and answers will appear here"
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Enter your question",
            placeholder="Type your question here...",
            scale=9,
            container=False
        )
        submit = gr.Button("Submit", variant="primary", scale=1)
    
    with gr.Row():
        clear = gr.Button("Clear Conversation", variant="secondary")
    
    gr.Markdown("### Sample Questions")
    gr.Markdown("Click any question below to ask it:")
    
    with gr.Column():
        for question in sample_questions:
            btn = gr.Button(question, size="sm")
            btn.click(lambda q=question: q, None, msg)
    
    msg.submit(chat, [msg, chatbot], [chatbot]).then(
        lambda: "", None, msg
    )
    submit.click(chat, [msg, chatbot], [chatbot]).then(
        lambda: "", None, msg
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    print("Starting Research Papers Q&A System")
    print("Local URL: http://127.0.0.1:7860")
    print("Model: Llama 3.2 3B")
    print("Running on GPU")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
    