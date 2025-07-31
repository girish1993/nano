from typing import Tuple

from picture import Picture
from rotate_transform import RotateTransform
from scaling_transform import ScalingTransform
from shear_transform import ShearTransform


class ImageTransformer:
    _TRANSFROM_MAPPING = {
        "scale": ScalingTransform,
        "rotate": RotateTransform,
        "shear": ShearTransform,
    }

    def __init__(
        self,
        obj: Picture,
        transform_type: str,
        tranform_factor: Tuple[int, int] | float | Tuple[float, float],
    ) -> None:
        self.transform_instance = ImageTransformer._TRANSFROM_MAPPING.get(
            transform_type
        )(obj)
        self.transform_factor = tranform_factor

    def transform(self):
        return self.transform_instance.transform(self.transform_factor)


if __name__ == "__main__":
    pic = Picture()
    pic.read_image(img_path="nano/assets/sample.jpg")

    Picture.plot(img=pic.img)
    transform_factor = (0, 0.25)
    # transform_factor_th = np.pi / 4

    img_trnsfrmr = ImageTransformer(
        obj=pic, transform_type="shear", tranform_factor=transform_factor
    )

    transfromed_img = img_trnsfrmr.transform()
    Picture.plot(img=transfromed_img, title="sheared transformation")
