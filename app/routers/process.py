from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from app.services import LLMService

router = APIRouter()
llm_service = LLMService()

@router.post("/upload-files")
async def upload_files(
    conversation_id: int,
    text_docs: str = None,
    files: List[UploadFile] = File(None)
):
    try:
        image_docs = []
        pdf_docs = []
        
        for file in files:
            content = await file.read()
            if file.content_type in ["image/jpeg", "image/png"]:
                image_docs.append(content)
            elif file.content_type == "application/pdf":
                pdf_docs.append(content)

        # Add documents to the graph
        llm_service.add_to_graph(
            conversation_id=conversation_id,
            text_docs=text_docs,
            image_docs=image_docs if image_docs else None,
            pdf_docs=pdf_docs if pdf_docs else None
        )
        
        return {"message": "Files processed and added to the graph successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@router.get("/query")
async def query_graph(conversation_id: int, query: str):
    try:
        result = llm_service.query(conversation_id, query)
        return {"result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while querying: {e}")
