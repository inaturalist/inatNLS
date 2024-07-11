import logging
import datetime
import json
from flask import request
logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class SearchService:
    def __init__(self, config, embedding_model, es_manager):
        self.config = config
        self.embedding_model = embedding_model
        self.es_manager = es_manager

    def perform_search(self, page, per_page, query, taxon_id, normalize_vectors=False):
        before_query_timestamp = datetime.datetime.now()
        logger.info(
            "search query: \'{}\' taxon_id: \'{}\' page: \'{}\' per_page: \'{}\'".format(
                query, taxon_id, page, per_page
            )
        )
        query_vector = self.embedding_model.get_embedding(
            query, normalize_vectors=normalize_vectors
        )
        filters = self.build_filters(taxon_id)

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

        self.write_logstash(query, taxon_id, page, per_page, before_query_timestamp, query_time)

        return results

    def build_filters(self, taxon_id):
        filters = { "filter": [ ] }

        excluded_taxon_ids = self.config.get("EXCLUDED_TAXON_IDS", [])
        if len(excluded_taxon_ids) > 0:
            exclude_terms = { 
                "terms": {
                    "taxon_ids": excluded_taxon_ids 
                } 
            } 
        
            filters["filter"].append(
                {
                    "bool": {
                        "must_not": exclude_terms
                    }
                },
            )
        
        if taxon_id:
            filters["filter"].append(
                {
                    "term": {
                        "taxon_ids": {
                            "term": taxon_id
                        }
                    }
                }
            )
        return filters

    def write_logstash(self, query, taxon_id, page, per_page, start_datetime, query_time):
        request_time = round(query_time.total_seconds() * 1000)
        logstash_log = open("./log/logstash.log", "a")
        log_data = {"@timestamp": start_datetime.isoformat(),
                    "query": query,
                    "taxon_id": taxon_id,
                    "page": page,
                    "per_page": per_page,
                    "client_ip": request.access_route[0],
                    "duration": request_time}
        json.dump(log_data, logstash_log)
        logstash_log.write("\n")
        logstash_log.close()
