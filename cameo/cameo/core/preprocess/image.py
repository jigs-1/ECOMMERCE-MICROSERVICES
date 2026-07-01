from functools import lru_cache

from PIL import Image
import torchvision.transforms as T


@lru_cache(maxsize=8)
def build_image_transform(image_size: int = 224, normalize: bool = True):
    transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]
    if normalize:
        transforms.append(
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return T.Compose(transforms)


def preprocess_image(image: Image.Image, image_size: int = 224):
    """
    Resize + normalize an image into a torch tensor (C,H,W).
    """
    transform = build_image_transform(image_size=image_size)
    return transform(image.convert("RGB"))


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_from_path(path: str, image_size: int = 224):
    return preprocess_image(load_image(path), image_size=image_size)
