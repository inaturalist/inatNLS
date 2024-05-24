import logging
import datetime

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class SearchService:
    def __init__(self, config, embedding_model, es_manager):
        self.config = config
        self.embedding_model = embedding_model
        self.es_manager = es_manager

    def perform_search(self, page, per_page, query, taxon_id):
        logger.info(
            "search query: \"{}\" taxon_id: \"{}\"".format(
                query, taxon_id
            )
        )
        query_vector = self.embedding_model.get_embedding(query)
        filters = self.build_filters(
            taxon_id
        )

        before_query_timestamp = datetime.datetime.now()
        results = self.es_manager.search(
            index_name=self.config["ES_INDEX_NAME"],
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": self.config["KNN"]["K"],
                "num_candidates": self.config["KNN"]["NUM_CANDIDATES"],
                **filters,
            },
            size=per_page,
            from_=page * per_page,
        )
        after_query_timestamp = datetime.datetime.now()
        query_time = after_query_timestamp - before_query_timestamp
        logging.info("query time was {}".format(query_time))

        return results

    def build_filters(self, taxon_id):
        filters = {"filter": []}
        if taxon_id:
            filters["filter"].append(
                {"term": {"taxon_ids": {"term": taxon_id}}})
        return filters
