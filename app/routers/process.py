from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from app.services import LLMService
import nest_asyncio
import os
nest_asyncio.apply()
import asyncio

router = APIRouter()
llm_service = LLMService()

@router.post("/upload-files")
async def upload_files(
    conversation_id: int,
    text_docs: str = None,
    # files: List[UploadFile] = File(None)
):
    try:
        image_docs = []
        pdf_docs = []
        
        # for file in files:
        #     content = await file.read()
        #     if file.content_type in ["image/jpeg", "image/png"]:
        #         image_docs.append(content)
        #     elif file.content_type == "application/pdf":
        #         pdf_docs.append(content)
        text_docs = '''
        # In the Kitchen

Reviewed by Peter Gottlieb

Qi Chien understands how restaurants work. Her new book, In the Kitchen: How to Thrive in the Restaurant Business, expertly advises restaurant managers on handling picky diners. Throughout the book, Chien offers concise, practical suggestions with easy-to-understand concepts. Overall, the book offers a colorful snapshot of the challenges in the industry, from creating reasonable schedules for chefs to appeasing various tasks involved in the day-to-day operations of a restaurant.

Chien's book is unique among other industry guides in that she interviewed restaurant owners, managers, and customers as part of her research. She even spoke to journalists who write restaurant reviews to get a good sense of what they most prize in a dining experience. My only criticism is that the book should have also included the perspective of chefs, especially since their role is crucial to a restaurant’s success. This caveat aside, In the Kitchen is an insightful and instructive read.

To: editor@lakecountyherald.com

From: qichien@rapidonline.com

Date: August 5

Subject: In the Kitchen

# To the Editor:

Date: August 5

Newspaper:

I have appreciated his thoughtful comments about my works over the years, delighted to read Peter Gottlieb's review of my latest book, In the Kitchen, in your newspaper. I was especially glad that he liked the chapter about restaurant reviewers, since initially I had been reluctant to interview the journalists for the book. It is true that I could have included a greater variety of insights, but unfortunately the people whose views he most wanted to hear were just too busy to speak with me before the publishing deadline. Perhaps this is something I can address in an updated edition of the book.

Qi Chien
        
        '''
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
        # nest_asyncio.apply()
        print(query)

        # result = llm_service.query(conversation_id, query)
        # return {"result": result}
        result = await asyncio.to_thread(llm_service.query, conversation_id, query)
        return {"result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while querying: {e}")
