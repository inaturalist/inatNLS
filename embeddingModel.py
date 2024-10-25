import PIL
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoProcessor, AutoModel

# informal interface
class EmbeddingModel:
    def __init__(self, model_name):
        pass

    def get_embedding(self, input_data):
        pass

class STEmbeddingModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, input_data):
        return self.model.encode(input_data)

# eg google/siglip-base-patch16-224
class SiglipEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)


    def get_embedding(self, input_data):
        if isinstance(input_data, str):
            return self.embed_text(input_data)
        elif isinstance(input_data, PIL.Image.Image):
            return self.embed_image(input_data)
        else:
            assert(f"no instance type for {input_data}")

    def embed_text(self, input_text):
        dummy = PIL.Image.new('RGB', (400, 400))
        inputs = self.processor(
            text=[ input_text ],
            images=[dummy],
            padding="max_length", 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.text_embeds[0].numpy()

    def embed_image(self, input_image):
        inputs = self.processor(
            images=[ input_image ],
            text=[ "dummy" ],
            padding="max_length", 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.image_embeds[0].numpy()

