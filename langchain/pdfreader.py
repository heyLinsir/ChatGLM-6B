from transformers import AutoTokenizer, AutoModel, pipeline
from langchain import VectorDBQA
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

chatglm_name_or_path = 'THUDM/chatglm-6b'
embedder_name_or_path = 'GanymedeNil/text2vec-large-chinese'
pdf_file_path = './data/nk29.pdf'
# query = "问答系统包含哪些主要模块？"
query = "ChatGPT的性能如何？"
chatglm_name_or_path = '/data/niuyilin/pre-trained-models/chatglm-6b'
embedder_name_or_path = '/data/niuyilin/pre-trained-models/text2vec-large-chinese'

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(chatglm_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(chatglm_name_or_path, trust_remote_code=True).float()
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2000
)
llm = HuggingFacePipeline(pipeline=pipe)

# 读取PDF文本内容
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()
for document in documents:
    document.page_content = document.page_content.replace('\n', '')
print('original document number:', len(documents))

# 分割长文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)
split_docs = text_splitter.split_documents(documents)
print('split document number:', len(split_docs))

# 加载句子编码模型 & 构建文本索引库
embeddings = HuggingFaceEmbeddings(
                    model_name=embedder_name_or_path,
                    model_kwargs={'device': 'cpu'}
                )
docsearch = Chroma.from_documents(split_docs, embeddings)

# 构建prompt
prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="根据文章内容回答问题。\n文章：{context}\n问题：{question}\n答案："
        )
# prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template="文章：\n{context}\n\n问题：\n{question}\n\n文章内容能否回答该问题，输出“能”或者“不能”，若能回答，则同时输出答案："
#         )
            # template="文章：\n{context}\n\n问题：\n{question}\n\n根据文章内容回答问题，若文章中未包含回答问题所需的相关信息，则回答“不知道”："

# 构建基于文本库的问答系统
qa = VectorDBQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            vectorstore=docsearch,
            k=5, # 检索得到的证据文本数量
            chain_type_kwargs={"prompt": prompt}
        )

# 生成答案
print(qa.run(query))
