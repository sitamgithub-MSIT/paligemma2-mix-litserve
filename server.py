# Import the required libraries
import os
from dotenv import load_dotenv
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import litserve as ls


# Load the Environment Variables from .env file
load_dotenv()

# Access token for using the model
access_token = os.environ.get("ACCESS_TOKEN")


class PaliGemma2MixAPI(ls.LitAPI):
    """
    PaliGemma2MixAPI is a subclass of ls.LitAPI that provides an interface to the finetuned PaliGemma2 Mix model.

    Methods:
        - setup(device): Initializes the model and processor with the specified device.
        - decode_request(request): Convert the request payload to model input.
        - predict(model_inputs): Uses the model to generate OCR text from the input image.
        - encode_response(output): Convert the model output to a response payload.
    """

    def setup(self, device):
        """
        Sets up the model and processor on the specified device.
        """
        model_id = "google/paligemma2-3b-mix-448"
        self.device = device
        self.model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, token=access_token
            )
            .eval()
            .to(self.device)
        )
        self.processor = PaliGemmaProcessor.from_pretrained(
            model_id, token=access_token
        )

    def decode_request(self, request):
        """
        Convert the request payload to model input.
        """
        # Extract the image path from the request
        image = load_image(request["image_path"])

        # Prepare the prompt for the OCR
        prompt = f"<image>ocr"

        # Prepare the model inputs
        return (
            self.processor(text=prompt, images=image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )

    def predict(self, model_inputs):
        """
        Run inference and get the model output.
        """
        input_len = model_inputs["input_ids"].shape[-1]

        # Run inference on the image to get the text output
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=256, do_sample=False
            )
            generation = generation[0][input_len:]
            return self.processor.decode(generation, skip_special_tokens=True)

    def encode_response(self, output):
        """
        Convert the model output to a response payload.
        """
        return {"text": output}


if __name__ == "__main__":
    # Create an instance of the PaliGemma2MixAPI class and run the server
    api = PaliGemma2MixAPI()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
