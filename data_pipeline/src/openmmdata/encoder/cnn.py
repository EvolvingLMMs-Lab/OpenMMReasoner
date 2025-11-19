import torch
from imagededup.methods import CNN


class CNNEncoder:
    def __init__(
        self,
    ):
        """
        Initializes the CNNEncoder with an optional model path.

        :param model_path: Path to the pre-trained CNN model. If None, uses the default model.
        """
        self.encoder = CNN()
        self.encoder.model = (
            self.encoder.model.eval()
        )  # Set the model to evaluation mode

    @torch.no_grad()
    def encode(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image into a feature vector using the CNN model.

        :param image_path: Path to the image file to be encoded.
        :return: A list representing the feature vector of the image.
        """
        return self.encoder.model(image_tensors)

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def device(self):
        return self.encoder.device
