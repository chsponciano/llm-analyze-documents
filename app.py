import os
import time
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

@st.cache_resource
def load_llm_model(temperature: float = 0.1):
    """
    Load the LLM model into memory in HuggingFace HUB
    """
    return HuggingFaceHub(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        model_kwargs={
            'temperature': temperature,
            'return_full_text': False,
            'max_new_tokens': 512,
        }
    )

def generate_retriever(uploads: list):
   """
   Generate retriever based on files
   """

   # Upload documents
   docs = []
   temp_dir = tempfile.TemporaryDirectory()
   
   for file in uploads:
      temp_filepath = os.path.join(temp_dir.name, file.name)
      with open(temp_filepath, 'wb') as f:
         f.write(file.getvalue())
      loader = PyPDFLoader(temp_filepath)
      docs.extend(loader.load())
    
   # Splitting into chunks of text / split
   text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
   splits = text_splitter.split_documents(docs)

   # Embedding
   embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-m3')

   # Storage
   vectorstore = FAISS.from_documents(splits, embeddings)
   vectorstore.save_local('vectorstore/db_faiss')

   # Retriever Setup
   return vectorstore.as_retriever(
        search_type = 'mmr', # The Maximal Marginal Relevance (MMR) criterion strives to reduce redundancy while maintaining query relevance in re-ranking retrieved documents
        search_kwargs={
            'k': 3, 
            'fetch_k': 4
        }
    )

def generate_chain(
    llm, 
    retriever,
    token_s='<|begin_of_text|><|start_header_id|>system<|end_header_id|>', 
    token_e='<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
):
    """
    Connects the model with the prompts generating the chain
    """
    
    # Contextualization prompt
    # (query, chat history) -> LLM -> reformulated query -> retriever
    context_q_system_prompt = 'Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. ALWAYS ANSWER IN BRAZILIAN PORTUGUESE.'
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = 'Question: {input}' + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', context_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', context_q_user_prompt),
        ]
    )

    # Chain for contextualization
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=context_q_prompt
    )
    
    # Prompt for questions and answers (Q&A)
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais. 
    Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa. 
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configure LLM and Chain for Questions and Answers (Q&A)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain,)

if __name__ == '__main__':
    # Loading Environment Variables
    load_dotenv()

    # Streamlit Settings
    st.set_page_config(page_title='Converse com documentos 📚', page_icon='📚')
    st.title('Converse com documentos 📚')

    # Loading LLM model in cache
    llm = load_llm_model()

    # Creating a side panel in the interface
    uploads = st.sidebar.file_uploader(
        label = 'Enviar arquivos', 
        type=['pdf'],
        accept_multiple_files=True
    )

    # Creating state variables
    if 'initialized' not in st.session_state:
        st.session_state.docs_list = None
        st.session_state.retriever = None
        st.session_state.chat_history = [
            AIMessage(content='Olá, sou o seu assistente virtual! Como posso ajudar você?'),
        ]
        st.session_state.initialized = True

    # Validate if not to any file
    if not uploads:
        st.info('Por favor, envie algum arquivo para continuar')
        st.stop()

    # Validates if the list of files is different from the upload
    elif st.session_state.docs_list != uploads:
        with st.spinner('Carregando os documentos...'):
            start = time.time()
            st.session_state.docs_list = uploads
            st.session_state.retriever = generate_retriever(uploads)
            end = time.time()
            print('Tempo para gerar o retriever: ', end - start)

    # Displays messages on the screen based on history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
                
        elif isinstance(message, HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)

    start = time.time()
    user_query = st.chat_input('Digite sua mensagem aqui...')

    if user_query is not None and user_query != '' and uploads is not None:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message('Human'):
            st.markdown(user_query)

        with st.chat_message('AI'):
            rag_chain = generate_chain(llm, st.session_state.retriever)
            result = rag_chain.invoke({'input': user_query, 'chat_history': st.session_state.chat_history})
            answer = result['answer']
            st.write(answer)

            sources = result['context']
            for idx, doc in enumerate(sources):
                source = doc.metadata['source']
                file = os.path.basename(source)
                page = doc.metadata.get('page', 'Página não especificada')

                ref = f':link: Fonte {idx}: *{file} - p. {page}*'
                with st.popover(ref):
                    st.caption(doc.page_content)

        st.session_state.chat_history.append(AIMessage(content=answer))

    end = time.time()
    print('Tempo para gerar a resposta: ', end - start)