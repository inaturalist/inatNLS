import csv
import hashlib
import logging
import os
from pathlib import Path
import yaml

from elasticsearch import Elasticsearch
import numpy as np
from PIL import Image
import requests
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

CONFIG = yaml.safe_load(open("config.yml"))


class Search:
    def __init__(self):
        self.model = SentenceTransformer(CONFIG["model_name"])
        self.es = Elasticsearch(CONFIG["es_url"])
        self.image_cache_path = Path(CONFIG["image_cache_dir"])
        self.max_workers = CONFIG["insert_max_workers"]
        os.makedirs(self.image_cache_path, exist_ok=True)
        client_info = self.es.info()
        print("Connected to Elasticsearch!")

    def get_embedding(self, text):
        return self.model.encode(text)

    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name, ignore_unavailable=True)

    def create_index(self, index_name):
        self.es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "ancestry" : {
                        "type" : "keyword"
                    },
                    "continent" : {
                        "type" : "keyword"
                    },
                    "country_name" : {
                        "type" : "keyword"
                    },
                    "embedding": {
                        "type": "dense_vector",
                    },
                    "extension" : {
                        "type" : "keyword"
                    },
                    "observation_uuid" : {
                        "type" : "keyword"
                    },
                    "observed_on" : {
                        "type" : "date"
                    },
                    "observer_id" : {
                        "type" : "keyword"
                    },
                    "observer_login" : {
                        "type" : "keyword"
                    },
                    "photo_id" : {
                        "type" : "keyword"
                    },
                    "quality_grade" : {
                        "type" : "keyword"
                    },
                    "taxon_id" : {
                        "type" : "keyword"
                    },
                    "taxon_name" : {
                        "type" : "text"
                    }
                }
            },
        )

    def download_and_process_image(self, document, index_name, local_path, img_emb_function, pbar):
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Attempt download only if the image file doesn't exist
            if not os.path.exists(local_path):
                image_base_url = (
                    "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/medium.{}"
                )
                photo_url = image_base_url.format(document["photo_id"], document["extension"])

                r = requests.get(photo_url)
                if r.status_code == 200:
                    with open(local_path, "wb") as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                else:
                    # Handle unsuccessful download (e.g., log the error)
                    print(f"Failed to download image from {photo_url}")
                    return None

            # Open the image and generate embedding
            if os.path.exists(local_path):
                img = Image.open(local_path)
                img_emb = img_emb_function(img)
                return {"index": {"_index": index_name}}, {**document, "embedding": img_emb}
            else:
                # Handle missing image locally (e.g., log the error)
                print(f"Image not found locally: {local_path}")
                return None

        except Exception as e:
            # Handle critical errors during processing (e.g., log the error)
            print(f"Error processing document {document['photo_id']}: {e}")
            return None

    def insert_documents(self, documents, index_name, pbar):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepare operations in advance
            futures = []
            for document in documents:
                local_path = self.path_for_photo_id(
                    CONFIG["image_cache_dir"], document["photo_id"]
                )
                future = executor.submit(
                    self.download_and_process_image,
                    document,
                    index_name,
                    local_path,
                    self.get_embedding,
                    pbar,
                )
                futures.append(future)

            # Collect results and update progress bar
            operations = []
            for future in futures:
                result = future.result()
                if result:
                    operations.extend(result)
                    pbar.update(1)  # Update progress bar for each successful operation
            
            self.es.bulk(index=index_name, operations=operations)

    def add_to_index(self, index_name, data_file):
        with open(data_file, "r") as file:
            num_docs = len(file.readlines()) - 1

        pbar = tqdm(
            total=num_docs,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            dynamic_ncols=True,
        )
        with open(data_file) as csvfile:
            csvreader = csv.DictReader(csvfile)
            documents = []
            for row in csvreader:
                documents.append(row)

                # insert in batches
                if len(documents) == CONFIG["insert_batch_size"]:
                    self.insert_documents(documents, index_name, pbar)
                    documents = []

        # insert anything at the end
        response = self.insert_documents(documents, index_name, pbar)

        pbar.close()

    def search(self, index_name, **query_args):
        return self.es.search(index=index_name, **query_args)

    def path_for_photo_id(self, base_dir, photo_id):
        msg = str(photo_id).encode("utf-8")
        m = hashlib.sha256()
        m.update(msg)
        hashed = m.hexdigest()
        filename = "{}.jpg".format(photo_id)
        photo_paths = [base_dir, hashed[0:2], hashed[2:4], filename]
        return os.path.join(*photo_paths)
