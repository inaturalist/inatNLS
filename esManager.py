import logging

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class ElasticSearchManager:
    def __init__(self, url="http://localhost:9200"):
        self.es = Elasticsearch(url, timeout=30)
        logger.info("Connected to Elasticsearch!")

    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name, ignore_unavailable=True)

    def create_index(self, index_name):
        self.es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "index": True,
                        "similarity": "cosine",
                        "dims": 512
                    }
                }
            },
        )

    def search(self, index_name, **query_args):
        return self.es.search(index=index_name, **query_args)

    def index_item(self, index_name, document):
        return self.es.index(index=index_name, document=document)

    def bulk_insert(self, index_name, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': index_name}})
            operations.append(document)
        return self.es.bulk(operations=operations)
