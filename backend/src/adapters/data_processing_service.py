import uuid
from typing_extensions import override
import fitz
import pickle
import os
import io
import base64
import numpy as np
import logging
from PIL import Image as PILImage
from tqdm import tqdm
from domain.context import Context
from domain.file_data import FileData
from domain.knowledge import FormattedKnowledge, Knowledge
from ports.data_processing_service_port import DataProcessingServicePort
from ports.data_repository_service_port import DataRepositoryServicePort
from ports.inference_service_port import InferenceServicePort
from ports.retrieval_service_port import RetrievalServicePort
from langchain.text_splitter import RecursiveCharacterTextSplitter
from domain.image import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessingService(DataProcessingServicePort):
    _retrieval_service: RetrievalServicePort
    _chunk_size: int

    def __init__(
        self,
        retrieval_service: RetrievalServicePort,
        data_repository_service: DataRepositoryServicePort,
        inference_service: InferenceServicePort,
        chunk_size: int,
        embedding_model_name: str,
        embedding_dimension_size: int,
        chunk_overlap: int,
        pdf_folder: str,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._data_repository_service = data_repository_service
        self._inference_service = inference_service
        self._chunk_size = chunk_size
        self._embedding_model_name = embedding_model_name
        self._embedding_dimension_size = embedding_dimension_size
        self._chunk_overlap = chunk_overlap
        self._pdf_folder = pdf_folder

    @override
    def generate_knowledge(self) -> list[Knowledge]:
        """
        Entry point, which includes all the different document processing pipes. 
        Currently just PDF processing
        """
        pdf_knowledge = self._generate_pdf_knowledge()
        logger.info(f"Generated {len(pdf_knowledge)} embedding objects")

        return pdf_knowledge
    
    def _generate_pdf_knowledge(self):
        pdf_list: list[FileData] = self._data_repository_service.load_file_data(
            file_type="pdf"
        )

        logger.info(f"Loaded {len(pdf_list)} pdfs")

        # Process the PDF bytes
        documents: list[FileData] = []
        all_images: dict[int, Image] = {}  # Temporary storage for all images with image_counter as key
        image_counter: int = 0

        # Process PDF documents
        for pdf in pdf_list:
            if not pdf.file_bytes:
                raise ValueError("ERROR: File bytes are empty.")

            text, doc_images, image_counter = self._process_pdf_bytes(
                pdf.file_bytes, image_counter
            )
            documents.append(FileData.create(file_bytes=None, file_text=text, source=pdf.source))
            all_images.update(doc_images)

        # Split the documents into defined chunks
        chunks: list[FileData] = self._split_documents(documents=documents)

        logger.info("Converting to indexable knowledge objects")
        # Generate the embedding objects for ingestion
        pdf_knowledge: list[Knowledge] = self._convert_chunks_to_knowledge(
            chunks=chunks
        )

        return pdf_knowledge
    

    @override
    def format_knowledge(self, docs: list[Knowledge], sensor_data: str) -> FormattedKnowledge:
        """Combine question with contexts and their metadata into a single string."""
        formatted_contexts: list[Context] = []
        document_sources: list[str] = []
        for knowledge in docs:
            metadata_format = self._format_metadata(knowledge)

            formatted_context = self._format_context_and_metadata(
                knowledge.content, metadata_format, knowledge.source
            )

            formatted_contexts.append(Context.create(formatted_context))
            document_sources.append(f"\nSource for this context: {knowledge.source}")
        formatted_contexts.append(Context.create(f"\n<sensor data> {sensor_data} </sensor data>"))
        logger.info("Formatted knowledge")
        return FormattedKnowledge.create(
            contexts=formatted_contexts, sources=document_sources
        )  # Return chunk images here if needed


    def _load_cache_from_file(self, cache_file: str) -> dict[str, str | bool]:
        # Load cache from file if it exists
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                result: dict[str, str | bool] = pickle.load(file)
                logger.info("Loaded cache")
                return result
        return {}

    def _save_cache_to_file(self, cache: dict[str, str | bool], cache_file: str):
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as file:
            pickle.dump(cache, file)
            logger.info("Saved cache")



    def _convert_chunks_to_knowledge(
        self,
        chunks: list[FileData],
    ) -> list[Knowledge]:
        """Takes in chunks, then creates a list of knowledge out of them with the corrent data format."""
        total_knowledge: list[Knowledge] = []
        for chunk in tqdm(chunks, total=len(chunks)):
            file_text = chunk.file_text
            if not file_text:
                raise ValueError("ERROR: File text is empty.")

            knowledge: Knowledge = self._build_knowledge_object(
                text_content=file_text, source=chunk.source
            )            

            total_knowledge.append(knowledge)

        logger.info("Converted context chunks to knowledge")
        return total_knowledge
    
    def _build_knowledge_object(self, text_content: str, source: str) -> Knowledge:
        """Builds a knowledge object from the given data."""
        # Generate a unique identifier for the document chunk
        doc_id = str(uuid.uuid4())  # Ensure each document has a unique ID

        text_content_embedding: list[float] = self._retrieval_service.embed_text(
            text=text_content
        )

        knowledge = Knowledge.create(
            id=doc_id,
            content=text_content,
            embedding=text_content_embedding,
            chunk_size=self._chunk_size,
            source=source,
        )

        logger.info("Built knowledge objects from text and metadata")
        return knowledge


    def _split_documents(self, documents: list[FileData]) -> list[FileData]:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        split_texts: list[FileData] = []

        for document in tqdm(documents, total=len(documents)):
            source = str(document.source)

            # Split the preprocessed text into chunks
            if document.file_text is None:
                continue    
            text_chunks: list[str] = text_splitter.split_text(document.file_text)

            # Create FileData objects for each chunk, retaining the source information
            for chunk in text_chunks:
                file_data = FileData.create(
                    file_bytes=None, file_text=chunk, source=source
                )
                split_texts.append(file_data)
        
        logger.info("Split documents into chunks")
        return split_texts


    def _format_metadata(self, context: Knowledge) -> str:
        """
        Placeholder for metadata formatting into a cohesive string.
        """

        logger.info("Formatted metadata")
        return ""

    def _format_context_and_metadata(
        self, context: str, metadata_format: str, source: str
    ) -> str:
        """Formats the context with metadata."""
        formatted_context = (
            f"\n<Start Metadata>\n{metadata_format}\n<End Metadata>"
            f"\n<Start Context>\n{context}\n<End Context>"
            f"\n<Start Source>\n{source}\n<End Source>"  
        )
        logger.info("Formatted contexts with metadata")
        return formatted_context

    def _process_pdf_bytes(
        self, pdf_bytes: bytes, image_counter: int
    ) -> tuple[str, dict[int, Image], int]:
        """
        Processes PDF bytes in order to pinpoint the location of the images in the text.
        Images are currently not leveraged other than storage, but can be interpreted for extra contextual information.
        """
        # Open the PDF in PyMuPDF (fitz) using the bytes stream
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        image_data: dict[int, Image] = {}
        full_text_with_placeholders = []

        for page in doc:
            # Extract words with their positions
            words: list[tuple[float, float, float, float, str, int, int, int]] = (
                page.get_text("words")
            )  # List of tuples: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            words.sort(key=lambda w: (w[1], w[0]))  # Sort by y0 (top), then x0 (left)

            text: str = ""
            word_positions: list[tuple[float, float, float, float, int]] = (
                []
            )  # To store positions of words in the text

            # Build the text content and keep track of positions
            prev_y = None
            for w in words:
                x0, y0, x1, y1, word = w[:5]
                if (
                    prev_y and abs(y0 - prev_y) > 5
                ):  # New line if Y-coordinate changes significantly
                    text += "\n"
                elif text:
                    text += " "
                word_positions.append(
                    (float(x0), float(y0), float(x1), float(y1), len(text))
                )
                text += str(word)
                prev_y = float(y0)

            # Get images from the page
            images: list[any] = page.get_images(full=True)

            for img in images:
                xref = img[0]
                # Extract image bytes and metadata
                image_info = doc.extract_image(xref)
                image_bytes = image_info["image"]
                image_ext = image_info["ext"]

                # Check the validity of the image
                if image_bytes and self._is_valid_image(image_bytes):
                    try:
                        # Get image bounding box
                        image_bbox: fitz.Rect = page.get_image_bbox(img)
                        x0_img, y0_img, x1_img, y1_img = (
                            float(image_bbox.x0),
                            float(image_bbox.y0),
                            float(image_bbox.x1),
                            float(image_bbox.y1),
                        )

                        # Calculate the insertion position in text
                        insertion_pos = None
                        min_distance = float("inf")
                        for w in word_positions:
                            x0_w, y0_w, x1_w, y1_w, text_index = w
                            # Calculate vertical distance between image and word
                            distance = abs(y0_img - y0_w)
                            if distance < min_distance:
                                min_distance = distance
                                insertion_pos = text_index

                        if insertion_pos is None:
                            # If no words on the page, append at the end
                            insertion_pos = len(text)

                        # Create image placeholder
                        image_placeholder = f" [IMAGE:{image_counter}] "

                        # Insert placeholder into text at the determined position
                        text = (
                            text[:insertion_pos]
                            + image_placeholder
                            + text[insertion_pos:]
                        )

                        # Store image metadata
                        img_data = Image.create(
                            tag="[IMAGE:" + str(image_counter) + "]",
                            img_bytes=image_bytes,
                            img_format=image_ext,
                        )
                        image_data[image_counter] = img_data
                        image_counter += 1
                    except RuntimeError as e:
                        logger.error(f"Error processing image: {e}")
                        continue

            # Append the processed page text with image placeholders
            full_text_with_placeholders.append(text)

        # Combine all the text pages into a single document string
        full_text = "\n".join(full_text_with_placeholders)

        logger.info("Processed PDF bytes")
        return full_text, image_data, image_counter


    def _is_valid_image(self, img_bytes: str | bytes) -> bool:
        """
        Validate the image bytes, then check for uniformity.
        If uniform, its skipped.
        """
        try:
            # Decode base64 if necessary
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)

            pil_image = PILImage.open(io.BytesIO(img_bytes))
            pil_image.verify()  # Verify it's a valid image
    
            pil_image = PILImage.open(io.BytesIO(img_bytes))
            pil_image_downscaled = pil_image.resize((64, 64), resample=1) 

            img_array = np.array(pil_image_downscaled.convert("L"))  # Grayscale conversion

             # Uniformity check
            _, counts = np.unique(img_array, return_counts=True)    
            dominant_percentage: float = max(counts) / sum(counts)

            logger.info("Validated image")
            return dominant_percentage <= 0.9   # Is the image more than 90% uniform?
        except Exception as e:
            logger.info("Image not valid")
            return False