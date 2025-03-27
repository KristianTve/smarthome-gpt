import os
from typing_extensions import override
from domain.knowledge import Knowledge
from domain.file_data import FileData
from ports.data_repository_service_port import DataRepositoryServicePort
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataRepositoryService(DataRepositoryServicePort):
    def __init__(
        self,
        db_connection_string: str,
        embedding_dimension_size: int,
        collection_name: str,
        pdf_folder: str

    ):
        self.db_connection_string = db_connection_string
        self.embedding_dimension_size = embedding_dimension_size
        self.collection_name = collection_name
        self.conn = psycopg2.connect(db_connection_string)
        self.conn.autocommit = True
        self.pdf_folder = pdf_folder
        logger.info("Connected to PostgreSQL database")

    def _execute_query(self, query: str, params=None):
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    @override
    def upload_knowledge(self, knowledge: list[Knowledge]) -> None:
        batch_size = 200
        serialized_knowledge = [k.to_dict() for k in knowledge]

        insert_query = f"""
        INSERT INTO {self.collection_name} (id, content, source, content_vector)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content;
        """
        
        try:
            for i in range(0, len(serialized_knowledge), batch_size):
                batch = serialized_knowledge[i: i + batch_size]
                values = [
                    (
                        doc['id'],
                        doc['content'],
                        doc['source'],
                        np.array(doc['content_vector']).tolist(),
                    )
                    for doc in batch
                ]
                with self.conn.cursor() as cur:
                    execute_values(cur, insert_query, values)
                logger.info(f"Successfully ingested {len(batch)} documents")
            logger.info("Ingested embeddings successfully")
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise

    @override
    def initialize_data_structure(self) -> None:
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.collection_name} (
            id TEXT PRIMARY KEY,
            content TEXT,
            source TEXT,
            content_vector VECTOR(%s)
        );
        """
        try:
            self._execute_query(create_table_query, (self.embedding_dimension_size,))
            logger.info("Database structure initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database structure: {e}")
            raise

    def load_file_data(self, file_type: str) -> list[FileData]:
        file_data_list = []
        for root, _, files in os.walk(self.pdf_folder):
            for file in files:
                if file.endswith(file_type):
                    file_path = os.path.join(root, file)
                    logger.info(f"Attempting to load file: {file_path}")
                    try:
                        with open(file_path, "rb") as f:  # read as bytes
                            file_bytes = f.read()
                        file_data_list.append(FileData(file_bytes, None, source="local"))     # Source not currently applicable due to local storage
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        return file_data_list

    def load_single_file_data(self, file_path: str) -> FileData:
        """Loads a single file from the given file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return FileData(file_path, content)
        except Exception as e:
            raise IOError(f"Error loading {file_path}: {e}")

