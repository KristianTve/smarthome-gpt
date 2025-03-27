from typing import Any
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import override
import psycopg2
from domain.model import Model
from ports.retrieval_service_port import RetrievalServicePort
from domain.knowledge import Knowledge
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalService(RetrievalServicePort):

    def __init__(
        self,
        db_connection_string: str,
        collection_name: str,
        n_chunks_retrieved: int,
        embedding_api_version: str,
        embedding_model_name: Model,
    ) -> None:
        self.db_connection_string = db_connection_string
        self.collection_name = collection_name
        self.n_chunks_retrieved = n_chunks_retrieved
        self.embedding_api_version = embedding_api_version
        self.embedding_model_name = embedding_model_name
        self.conn = psycopg2.connect(db_connection_string)
        self._embedding_function: OpenAIEmbeddings = self._embedding_function()
        self.conn.autocommit = True

    @override
    def embed_text(self, text: str) -> list[float]:
        try:
            return self._embedding_function.embed_query(text)
        except Exception as e:
            raise ValueError("ERROR: Could not embed the text; Error:\n\n", e)

    @override
    def find_text_context(self, text: str) -> list[Knowledge]:
        query_embedding: list[float] = self.embed_text(text)

        result: object = self._get_relevant_documents(query_embedding)
        print(result)
        knowledge_list: list[Knowledge] = []

        for doc in result:
            id, content, source, _ = doc
            
            try:
                knowledge = Knowledge.create(
                    id=str(id),
                    content=content,
                    embedding=None,
                    chunk_size=None,
                    source=source,
                )
                knowledge_list.append(knowledge)
            except Exception as e:
                raise e
        logger.info("Found "+str(len(knowledge_list)) + " relevant contexts")
        return knowledge_list


    def _embedding_function(self) -> OpenAIEmbeddings:
        logger.info("Instantiated embedding function")
        return OpenAIEmbeddings(
            model=self.embedding_model_name.name,
            api_version=self.embedding_api_version,
        )

    def _get_relevant_documents(self, query_vector: list[float]) -> object:
        sql = f"""
        SELECT id, content, source, content_vector
        FROM {self.collection_name}
        ORDER BY content_vector <-> %s::vector
        LIMIT %s;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (np.array(query_vector).tolist(), self.n_chunks_retrieved))
                logger.info("Queried the vector database for relevant documents")
                return cur.fetchall()
        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}")
            return []

    @override
    def retrieve_sensor_data(self) -> object:
        # Mocked for now:
        mocked_data = {
            "timestamp": "2025-03-24T12:00:00Z",
            "device_id": "AT-123456",
            "location": "Stue",
            "sensors": {
                "temperature": {
                "value": 21.5,
                "unit": "°C"
                },
                "humidity": {
                "value": 45,
                "unit": "%"
                },
                "co2": {
                "value": 620,
                "unit": "ppm"
                },
                "voc": {
                "value": 200,
                "unit": "ppb"
                },
                "radon": {
                "value": 30,
                "unit": "Bq/m³"
                },
                "pressure": {
                "value": 1012,
                "unit": "hPa"
                },
                "pm2_5": {
                "value": 8,
                "unit": "µg/m³"
                }
            },
            "battery": {
                "level": 85,
                "status": "Normal"
            },
            "status": "OK"
            }
        logger.info("Fetched sensor data")
        return mocked_data