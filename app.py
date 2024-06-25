import click
from flask import Flask, render_template, abort, request
import logging
from logging.handlers import RotatingFileHandler

from config import Config
from esManager import ElasticSearchManager
from imageManager import ImageManager
from ingestionServiceMultiThread import IngestionServiceMultiThread
from ingestionService import IngestionService
from embeddingModel import EmbeddingModel
from searchService import SearchService
from humanDetectionModel import HumanDetectionModel


def create_app():
    app = Flask(__name__)
    app_config = Config.load_config()
    app.config.update(app_config)

    embedding_model = EmbeddingModel(
        app.config["CLIP_MODEL_NAME"]
    )
    es_manager = ElasticSearchManager(url=app.config["ES_URL"])
    image_manager = ImageManager(cache_dir=app.config["IMAGE_CACHE_DIR"])
    human_detection_model = HumanDetectionModel(
        model_path=app.config["MEDIAPIPE_HUMAN_DETECT_MODEL"],
        threshold=app.config["HUMAN_EXCLUSION_THRESHOLD"]
    )

    search_service = SearchService(
        app.config,
        embedding_model=embedding_model,
        es_manager=es_manager,
    )

    ingestion_service = IngestionService(
        embedding_model=embedding_model,
        es_manager=es_manager,
        image_manager=image_manager,
        human_detection_model=human_detection_model,
    )
    app.search_service = search_service
    app.ingestion_service = ingestion_service

    # Create app logger
    file_handler = RotatingFileHandler(
        './log/app.log',
        maxBytes=1024 * 1024 * 100,
        backupCount=10
    )
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    return app


app = create_app()


@app.get("/")
def index():
    return render_template(
        "index.html",
    )


@app.post("/")
def handle_search():
    try:
        query = request.values.get("query", "")

        taxon_id = request.values.get("taxon_id", None)
        if taxon_id is not None:
            try:
                taxon_id = int(taxon_id)
            except ValueError:
                taxon_id = None

        page = request.values.get("page", 0)
        try:
            page = int(page)
        except ValueError:
            page = 0

        per_page = request.values.get("per_page", app.config.get("PER_PAGE_DEFAULT"))
        try:
            per_page = int(per_page)
        except ValueError:
            per_page = app.config.get("PER_PAGE_DEFAULT")

        results = app.search_service.perform_search(
            page, per_page, query, taxon_id
        )
        return {
            "page": page,
            "per_page": per_page,
            "total_results": app.config["KNN"]["K"],
            "results": [
                {
                    "photo_id": hit["_source"]["photo_id"],
                    "score": hit["_score"],
                }
                for hit in results["hits"]["hits"]
            ]
        }
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        abort(500, description="Internal Server Error")

@app.get("/status")
def status():
    return "nls-demo OK"


@app.cli.command()
@click.argument("filename", required=True)
def reindex(filename):
    """Add new data to elasticsearch index."""
    app.ingestion_service.ingest_data(
        filename,
        index_name=app.config.get("ES_INDEX_NAME"),
        ingestion_batch_size=app.config.get("INSERT_BATCH_SIZE"),
        ingestion_cap=app.config.get("INGESTION_CAP")
    )
