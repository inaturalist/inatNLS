import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class IngestionServiceMultiThread:
    def __init__(
        self, image_manager, es_manager, embedding_model, human_detection_model
    ):
        self.image_manager = image_manager
        self.es_manager = es_manager
        self.embedding_model = embedding_model
        self.human_detection_model = human_detection_model

    def download_and_process_image(self, row, local_path, photo_url, pbar):
        if not os.path.exists(local_path):
            if not self.image_manager.download_image(photo_url, local_path):
                logger.warn("can't download {}, skipping {}".format(
                    photo_url, local_path
                ))
                return None

        try:
            # exclude photos where we can find a human face
            if self.human_detection_model.detect_faces(local_path):
                logger.info("detected human face in {} above threshold, skipping.".format(
                    local_path)
                )
                return None

            img = self.image_manager.open_image(local_path)
            img_emb = self.embedding_model.get_embedding(img)
            document = {
                **row,
                "embedding": img_emb,
            }
            return document
        except Exception as e:
            logger.error(
                "couldn't open or encode {}: {}".format(
                    local_path, repr(e)
                )
            )
            return None

    def insert_documents(self, index_name, rows, pbar):
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Prepare operations in advance
            futures = []
            for row in rows:
                local_path = self.image_manager.path_for_photo_id(
                    row["photo_id"]
                )
                photo_url = self.image_manager.url_for_photo_id(
                    row["photo_id"], row["extension"]
                )

                future = executor.submit(
                    self.download_and_process_image,
                    row,
                    local_path,
                    photo_url,
                    pbar,
                )

                futures.append(future)

            # Collect results and update progress bar
            docs = []
            for future in futures:
                result = future.result()
                if result:
                    docs.append(result)
                    pbar.update(1)  # Update progress bar for each successful operation

            self.es_manager.bulk_insert(
                index_name=index_name, documents=docs
            )

    def ingest_data(self, data_file, index_name):
        self.es_manager.delete_index(index_name)
        self.es_manager.create_index(index_name)

        with open(data_file, "r") as file:
            num_docs = len(file.readlines()) - 1

        pbar = tqdm(
            total=num_docs,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            dynamic_ncols=True,
        )
        with open(data_file) as csvfile:
            csvreader = csv.DictReader(csvfile)
            batch = []
            for row in csvreader:
                batch.append(row)

                # insert in batches
                if len(batch) == 200:
                    self.insert_documents(index_name, batch, pbar)
                    batch = []

        # insert anything at the end
        self.insert_documents(index_name, batch, pbar)

        pbar.close()
