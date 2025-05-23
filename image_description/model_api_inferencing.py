import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import sys
import re
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from enigmaai import util

class ImageCaptionGenerator:
    def __init__(
        self, model_dir: str, max_new_tokens: int=50, num_beams: int=3, length_penalty: float=0.7, min_length: int=10, no_repeat_ngram_size: int=3):
        self.device = util.get_device_name()
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        # configure generation
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.min_length = min_length
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def generate_caption(self, image_path: str, prompt_text: str = "Describe the scene in this image: ", max_new_tokens: int = None, num_beams: int = None,
        return_tokens: bool = False):
        self.model.eval()
        img = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        # tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt")
        prompt_ids = prompt_inputs.input_ids.to(self.device)
        prompt_len = prompt_ids.size(-1)

        gen_kwargs = {
            "pixel_values": pixel_values,
            "decoder_input_ids": prompt_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.max_new_tokens,
            "min_length": self.min_length,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            }

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        # slice away the prompt tokens
        gen_ids = output_ids[0][prompt_len:]
        if return_tokens:
            return gen_ids
        # decode and strip any leading junk
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned = re.sub(r'^[^A-Za-z0-9]+', '', raw).strip()
        if "." in cleaned:
            cleaned = cleaned[: cleaned.rfind(".") + 1 ]
        return cleaned

if __name__=="__main__":
    # instantiate once
    generator = ImageCaptionGenerator(model_dir="C:\\Users\\kamat\\OneDrive\\Documents\\GitHub\\AIStudio\\EnigmaAI\\image_description\\models\\best_model")

    # generate a caption
    caption = generator.generate_caption(
        image_path="C:\\Users\\kamat\\OneDrive\\Documents\\GitHub\\AIStudio\\EnigmaAI\\image_description\\models\\frame_IMG_4329_00010.jpg",
        prompt_text="Summarize the scene in this image:")
    print(caption)

