from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class RAGModelHandler:
    def __init__(self):
        print("Loading model... This might take a while.")
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="exact"
        )
        self.model = RagTokenForGeneration.from_pretrained(
            "facebook/rag-token-nq", retriever=self.retriever
        )
        print("Model loaded successfully.")

    def generate_response(self, query: str, max_length: int = 30):
        print(f"Generating response for query: {query}")
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=max_length)
        print(f"Generated response: {output[0]}")
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class GenerateRequest(BaseModel):
    query: str
    max_length: int = 30


app = FastAPI()
model_handler = RAGModelHandler()


# Generate endpoint for the API
@app.post("/generate", response_model=GenerateRequest, status_code=200)
async def generate(request: GenerateRequest):
    try:
        response = model_handler.generate_response(request.query, request.max_length)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Generate a path for / and return a simple message
@app.get("/", status_code=200)
async def root():
    print("Root path called.")
    return {"message": "Welcome to Token-RAG API"}
