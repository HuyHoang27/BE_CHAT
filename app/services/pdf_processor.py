from . import ProcessorBase
from .dependencies import (
    List,
    Document,
    LlamaParse,
    os,
    subprocess,
    re
)
class PdfProcessor(ProcessorBase):
    def process(self, data: List[bytes], conversation_id: str) -> List[Document]:
        sub_docs = self.process_pdf(data, conversation_id)
        return sub_docs
    def process_pdf(self, data: List[bytes], conversation_id: str) -> List[Document]:
        sub_docs = []
        for idx, pdf_data in enumerate(data):
            output_pdf = "./extract_pdf.pdf"
            input_pdf = "./input_pdf.pdf"
            with open(input_pdf, "wb") as f:
                f.write(pdf_data)
            self.run_ocrmypdf(input_pdf, output_pdf)
            
            parser_gpt4o = LlamaParse(
                result_type="markdown",
                parsing_instructions='This is a TOEIC test in Part 7. Please extract the correct format of Part 7 and fully extract all the questions and content included in the test.',
                gpt4o_mode=True,
                split_by_page=True,
            )
            documents_gpt4o = parser_gpt4o.load_data(output_pdf)
            final_paragraphs_save = self.extract_context_and_question(documents_gpt4o)
            os.remove(output_pdf)
            os.remove(input_pdf)
            for i in final_paragraphs_save:
                coref_text = self.graph.text_coref(i['Context'])
                chunks = self.graph.semantic_chunking(coref_text, 51)
                sub_docs_child = [Document(text=chunk, metadata={"conversation_id": conversation_id}) for chunk in chunks]
                sub_docs.extend(sub_docs_child)
        return sub_docs
    def run_ocrmypdf(self, input_pdf: str, output_pdf: str):
        command = ["ocrmypdf", input_pdf, output_pdf]
        try:
            subprocess.run(command, check=True)
            # In thông báo khi hoàn tất
            print(f"OCR process completed successfully for {input_pdf}")
        except subprocess.CalledProcessError as e:
            # In ra lỗi nếu quá trình OCR thất bại
            print(f"Error during OCR process: {e}")
    def extract_context_and_question(self, documents_gpt4o):
        group_paragraphs_json = {}
        id_paragraph = 0
        past_id_non_question = []
        final_result_paragraphs = []
        pattern_question_block = (
            r"(\d{3})\.\s+(.+?)\n"        # Match three-digit question number and question text
            r"(?:\n\s*)?"                 # Optional line break and whitespace
            r"(?:\s*-?\s*\(A\)\s+(.+?)[ \t]*\n)"    # Match option A, optionally with a dash
            r"(?:\s*-?\s*\(B\)\s+(.+?)[ \t]*\n)"    # Match option B, optionally with a dash
            r"(?:\s*-?\s*\(C\)\s+(.+?)[ \t]*\n)"    # Match option C, optionally with a dash
            r"(?:\s*-?\s*\(D\)\s+(.+))"             # Match option D, optionally with a dash
        )

        pattern_directions = (
            r"(#?\s*PART\s+\d+)\n"                      # Match "PART" with optional "#" and number
            r"\n?\*\*Directions:\*\*\s+(.+)"            # Match "**Directions:**" followed by instructions
        )


        for i in range(len(documents_gpt4o)):
        
            content = documents_gpt4o[i].get_content()
        
            if i == 0:
                match = re.search(pattern_directions, content)
                content = content[match.end():]
                
            if i == len(documents_gpt4o) - 1:
                match = re.search('Stop! This is the end of the test', content)
                content = content[:match.start()]
                
            content = content.replace('**', '').replace('*', '').replace('#', '').replace('- ', '').replace('GO ON TO THE NEXT PAGE', '').strip()
        
            match = re.search(pattern_question_block, content)
            
            # Sử dụng re.findall để tìm tất cả các kết quả phù hợp
            matches = list(re.finditer(pattern_question_block, content, re.DOTALL))
        
            if matches:
                id_paragraph = i
                group_paragraphs_json[id_paragraph] = []
        
                if len(past_id_non_question) != 0:
                    for item in past_id_non_question:
                        group_paragraphs_json[id_paragraph].append(item)
                    past_id_non_question = []
                    
                group_paragraphs_json[id_paragraph].append(content)
                final_result_paragraphs.append(group_paragraphs_json[i])

            else:
                past_id_non_question.append(content)

        final_paragraphs_save = []
        
        count = 0
        
        for paragraphs in final_result_paragraphs:
            count += 1
        
            paragraph_final = "".join(paragraphs)
        
            matches = list(re.finditer(pattern_question_block, paragraph_final, re.DOTALL))
        
            first_match_pos = matches[0].start()
        
            context = paragraph_final[:first_match_pos].strip()
            questions = paragraph_final[first_match_pos:].strip()

            paragraph_json = {}
            paragraph_json['Context'] = context
        
            matches_question = list(re.finditer(pattern_question_block, questions, re.DOTALL))
        
            quesions_lst = []
        
            for i, match_ques in enumerate(matches_question):
                start_question = match_ques.start()
                
                if i < len(matches_question) - 1:
                    end_question = matches_question[i +1].start()
                    question_extract = questions[start_question:end_question]
                else:
                    question_extract=questions[start_question:]
                    
                quesions_lst.append(question_extract)
        
            paragraph_json['Question'] = quesions_lst
        
            final_paragraphs_save.append(paragraph_json)

        return final_paragraphs_save