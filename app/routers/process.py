from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from app.services import LLMService
import logging
import os

router = APIRouter()
llm_service = LLMService()

# Initialize logging
logger = logging.getLogger(__name__)

@router.post("/upload-files", response_model=dict)
async def upload_files(
    conversation_id: int = Form(..., description="Unique ID for the conversation"),
    text_docs: Optional[str] = Form(None, description="A message string"),
    files: Optional[List[UploadFile]] = File(None, description="A list of binary files")  # Allow files to be optional
) -> dict:
    try:
        # Xử lý trường hợp khi files là None
        image_docs = []
        pdf_docs = []
        if files:
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
        logger.error(f"Error processing files for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/query", response_model=dict)
async def query_graph(
    conversation_id: str = Form(..., description="Unique ID for the conversation"),
    query: Optional[str] = Form(None, description="A message string"),) -> dict:
    try:
        result = llm_service.query(conversation_id, query)
        return {"result": result}
    
    except Exception as e:
        logger.error(f"Error querying graph for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while querying: {str(e)}")
