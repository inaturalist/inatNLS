import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


class IngestionService:
    def __init__(
        self, image_manager, es_manager, embedding_model, human_detection_model
    ):
        self.image_manager = image_manager
        self.es_manager = es_manager
        self.embedding_model = embedding_model
        self.human_detection_model = human_detection_model

    def ingest_data(self, data_file, index_name, ingestion_batch_size=50, ingestion_cap=None):
        # because we can't upsert into elastic search
        # we'll create duplicates if we're not careful
        # so instead we just recreate the index every time
        # this is fine for a demo/prototype
        self.es_manager.delete_index(index_name)
        self.es_manager.create_index(index_name)

        with open(data_file, "r") as file:
            num_docs = len(file.readlines()) - 1

        df = pd.read_csv(data_file)

        # convert ancestry to a list of taxon ids
        df["ancestry_list"] = df.ancestry.str.split("/")
        df["summed_ancestry_list"] = df.groupby("photo_id").ancestry_list.sum()
        photos_with_ancestries = (
            df
            .groupby(["photo_id", "extension", "taxon_id"])
            .ancestry_list.sum()
        )
        docs = []
        num_ingested = 0
        for ((photo_id, extension, taxon_id), ancestry_items) in photos_with_ancestries.items():
            if ingestion_cap is not None and num_ingested >= ingestion_cap:
                print("reached ingestion cap with {} records.".format(
                    num_ingested
                ))
                break

            ancestry_items = [int(taxon_id) for taxon_id in ancestry_items]
            ancestry_items = list(set(ancestry_items))
            ancestry_items = [x for x in ancestry_items if x != 48460]
            taxon_ids = ancestry_items + [taxon_id]

            local_path = self.image_manager.path_for_photo_id(photo_id)

            if not os.path.exists(local_path):
                continue

            try:
                # exclude photos where we can find a human face
                if self.human_detection_model.detect_faces(local_path):
                    print("detected human face in {} above threshold, skipping.".format(local_path))
                    continue

                img = self.image_manager.open_image(local_path)
                img_emb = self.embedding_model.get_embedding(img)
                document = {
                    "photo_id": photo_id,
                    "taxon_ids": taxon_ids,
                    "embedding": img_emb,
                }
                docs.append(document)

                # self.es_manager.index_item(
                #     index_name=index_name, document=document
                # )

                if len(docs) % ingestion_batch_size == 0:
                    self.es_manager.bulk_insert(
                        index_name=index_name, documents=docs
                    )
                    num_ingested += len(docs)
                    print("ingestion job {} / {}".format(num_ingested, num_docs))
                    docs = []
            except Exception as e:
                logger.error(
                    "couldn't open or encode {}: {}".format(
                        local_path, repr(e)
                    )
                )
                continue
