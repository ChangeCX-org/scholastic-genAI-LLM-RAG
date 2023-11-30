from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class RAGModelHandler:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="wiki_dpr"
        )
        self.model = RagTokenForGeneration.from_pretrained(
            "facebook/rag-token-nq", retriever=self.retriever
        )

    def generate_response(self, query: str, max_length: int = 30):
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class GenerateRequest(BaseModel):
    query: str
    max_length: int = 30


app = FastAPI()
model_handler = RAGModelHandler()


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        response = model_handler.generate_response(request.query, request.max_length)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
