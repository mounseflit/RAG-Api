import time
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import chromadb
from file_processor import process_file
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from fastapi.responses import RedirectResponse
import uvicorn
import python_multipart
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app = FastAPI(title="RAG Chatbot with Multi-File Support")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False as using wildcard origins with credentials=True is not allowed
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)



# In-memory state (for demo; use persistent store for production)
processed_files = []
chat_history = []


# Initialize ChromaDB client
client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")


# Model options (for reference, can be used in frontend)
MODEL_TEXT_OPTIONS = {
    "IBM Granite": [
        "ibm/granite-3-2-8b-instruct",
        "ibm/granite-3-2b-instruct",
        "ibm/granite-3-3-8b-instruct",
        "ibm/granite-3-8b-instruct"
    ],
    "Meta (LLaMA)": [
        "meta-llama/llama-3-2-1b-instruct",
        "meta-llama/llama-3-2-3b-instruct",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-3-405b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    ],
    "Mistral": [
        "mistralai/mistral-large",
        "mistralai/mistral-medium-2505",
        "mistralai/mistral-small-3-1-24b-instruct-2503"
    ]
}

class ChatRequest(BaseModel):
    api_key: str
    project_id: str
    url: str
    model_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    elapsed: float
    tokens: int
    speed: float
    used_chunks: int
    chat_history: list

class FileSummary(BaseModel):
    file_name: str
    chunk_count: int


@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    new_processed = []
    for uploaded_file in files:
        file_name = uploaded_file.filename
        if file_name in [f["metadata"]["file_name"] for f in processed_files]:
            continue
        # Create temp dir to store files with original names
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the file with its original name in the temp directory
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(await uploaded_file.read())
            
            file_extension = file_name.split('.')[-1].lower()
            if file_extension in ['txt']:
                file_type = 'txt'
            elif file_extension in ['pdf']:
                file_type = 'pdf'
            elif file_extension in ['docx']:
                file_type = 'docx'
            elif file_extension in ['xlsx']:
                file_type = 'xlsx'
            elif file_extension in ['png', 'jpg', 'jpeg']:
                file_type = 'image'
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
            
            try:
                processed_data = process_file(file_path, file_type)
                if processed_data:
                    for item in processed_data:
                        doc_id = f"{item['metadata']['file_name']}_{item['metadata']['chunk_id']}"
                        collection.add(
                            documents=[item['content']],
                            metadatas=[item['metadata']],
                            ids=[doc_id]
                        )
                    processed_files.extend(processed_data)
                    new_processed.extend(processed_data)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing {file_name}: {str(e)}")

    return {"message": "Files processed successfully!", "processed_files": [f["metadata"]["file_name"] for f in new_processed]}



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # if not all([request.api_key, request.project_id, request.model_id]):
    #     raise HTTPException(status_code=400, detail="Missing required fields.")
    # if not processed_files:
    #     raise HTTPException(status_code=400, detail="No files processed.")
    chat_history.append({"role": "user", "content": request.message})
    try:
        results = collection.query(query_texts=[request.message], n_results=3)
        context = ""
        if results['documents'] and results['documents'][0]:
            context = "Based on the following information from your documents:\n\n"
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context += f"Document {i+1} ({metadata['file_name']}):\n{doc}\n\n"
        system_message = """ You are an intelligent RAG assistant from OCP Group, specialized in extracting precise information from uploaded documents.
        
        When responding:
        1. Only use information directly present in the provided document context
        2. If the context doesn't contain the information needed to answer the question, clearly state that
        3. Keep responses concise and professional, using bullet points when appropriate
        4. Use the User Language for all responses
        5. For numerical data, maintain accuracy from the original documents
        6. If the question is outside the scope of the provided documents, politely inform the user
        7. Always ask clarifying questions if the user's request is ambiguous about which document is being referred to
        8. If the user asks for information not in the documents, say: "I don't know" or "I cannot answer that based on the provided documents."
        9. Try to mention the specific document and section when providing answers smoothly in the answer.
        10. Maintain a professional tone and neutral status about the information provided.

        Your role is to help users find accurate information from their uploaded documents, not to make assumptions beyond what is explicitly stated."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{context}\n\nQuestion: {request.message}"}
        ]


        # Get credentials from request or fall back to environment variables
        load_dotenv()
        
        url = request.url or os.getenv("WATSONX_URL")
        api_key = request.api_key or os.getenv("WATSONX_API_KEY")
        model_id = request.model_id or os.getenv("WATSONX_MODEL_ID")
        project_id = request.project_id or os.getenv("WATSONX_PROJECT_ID")
        
        if not all([url, api_key, model_id, project_id]):
            raise HTTPException(status_code=400, detail="Missing credentials. Provide them in the request or set environment variables.")
            
        credentials = Credentials(url=url, api_key=api_key)
        params = TextChatParameters(temperature=0.7)
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params
        )
        
        start_time = time.time()
        response = model.chat(messages=messages)
        end_time = time.time()
        content = response["choices"][0]["message"]["content"]
        elapsed = end_time - start_time
        tokens = len(content.split())
        speed = tokens / elapsed if elapsed > 0 else 0
        chat_history.append({"role": "assistant", "content": content})
        num_chunks = len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0
        return ChatResponse(
            answer=content,
            elapsed=elapsed,
            tokens=tokens,
            speed=speed,
            used_chunks=num_chunks,
            chat_history=chat_history[-10:] # last 10 messages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/file_summary", response_model=List[FileSummary])
async def file_summary():
    summary = {}
    for item in processed_files:
        file_name = item["metadata"]["file_name"]
        if file_name not in summary:
            summary[file_name] = 0
        summary[file_name] += 1
    return [FileSummary(file_name=k, chunk_count=v) for k, v in summary.items()]

@app.post("/clear_chat")
async def clear_chat():
    chat_history.clear()
    return {"message": "Chat history cleared."}

@app.post("/clear_files")
async def clear_files():
    processed_files.clear()
    return {"message": "Processed files cleared."}

@app.get("/system_info")
async def system_info():
    return {"total_indexed_chunks": len(processed_files)}

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# Serve static files

# Mount the static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/ui")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "index.html not found in static directory"})

# # Render deployment needs access to the app directly
# # The port is automatically set by Render via the PORT environment variable
# if __name__ == "__main__":

#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)









