import click
from flask import Flask, render_template, request
import yaml
import logging
from logging.handlers import RotatingFileHandler

from config import Config
from data import iconic_taxa, continent_choices
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

    # Create logger
    file_handler = RotatingFileHandler('./logs/search.log', maxBytes=1024 * 1024 * 100, backupCount=10)
    formatter = logging.Formatter('[%(asctime)s]%(message)s')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    return app

app = create_app()


@app.get("/")
def index():
    return render_template(
        "index.html",
        continent_choices=continent_choices,
        iconic_taxa=iconic_taxa,
    )


@app.post("/")
def handle_search():    
    query = request.form.get("query", "")
    login = request.form.get("login", "")
    continent = request.form.get("continent", "")
    iconic_taxon = request.form.get("iconic_taxon", "")

    results = app.search_service.perform_search(
        query, login, continent, iconic_taxon
    )

    return render_template(
        "index.html",
        query=query,
        login=login,
        continent=continent,
        continent_choices=continent_choices,
        iconic_taxon=iconic_taxon,
        iconic_taxa=iconic_taxa,
        results=results["hits"]["hits"],
        from_=0,
        total=results["hits"]["total"]["value"],
    )

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
