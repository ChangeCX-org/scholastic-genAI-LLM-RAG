from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from fastapi import FastAPI
from rag_model_handler import RAGModelHandler


class RAGModelHandler:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="wiki_dpr"
        )
        self.model = RagTokenForGeneration.from_pretrained(
            "facebook/rag-token-nq", retriever=self.retriever
        )

    def generate_response(self, query, max_length=30):
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


app = FastAPI()
model_handler = RAGModelHandler()


@app.post("/generate")
async def generate(query: str, max_length: int = 30):
    return {"response": model_handler.generate_response(query, max_length)}
