import os
import os.path
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        is_train (bool): If True, creates a dataset from the training folder, otherwise from the test folder.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        idx_to_class (dict): Dict with items (class_index, class_name).`
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        is_train: bool = True,
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform

        # Load class information and create a dataset
        classes, class_to_idx, idx_to_class = self.find_classes(directory=self.root)
        samples = self._make_dataset(directory=self.root, class_to_idx=class_to_idx)
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        # TODO(hcchen): Add stratified sampling, default to 8:2 split
        train_samples, test_samples = train_test_split(
            samples, test_size=0.2, random_state=seed, stratify=[s[1] for s in samples]
        )
        samples = train_samples if is_train else test_samples

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def find_classes(
        self, directory: Union[str, Path]
    ) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int], Dict[int, str]]): List of all classes and dictionary mapping each class to an index and mapping an index to each class.
        """
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}
        return classes, class_to_idx, idx_to_class

    def _make_dataset(
        self, directory: str, class_to_idx: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        directory = os.path.expanduser(directory)
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")

        instances = []
        valid_extensions = (".jpg", ".jpeg", ".png")
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(valid_extensions):
                        path = os.path.join(root, fname)
                        instances.append((path, class_index))
        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
