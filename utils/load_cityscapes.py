from torchvision.datasets import Cityscapes
from typing import Optional, Sequence, Callable, Tuple
import torchvision.transforms.functional as TF
import torch
from collections import namedtuple
from torchvision.transforms import InterpolationMode, Normalize



# adapted from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )



class_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


class JointResize:
    def __init__(self, size_hw):
        self.size = size_hw  # (H, W)
    def __call__(self, img, mask):
        img  = TF.resize(img,  self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask.unsqueeze(0).float(), self.size, interpolation=InterpolationMode.NEAREST)\
                .squeeze(0).long()
        return img, mask



class CityscapesSubset(Cityscapes):
    """
    Cityscapes dataset wrapper that filters labels to a subset of classes and
    optionally remaps them to compact IDs.

    This class inherits from ``torchvision.datasets.Cityscapes`` and returns:
    - an RGB image tensor of shape ``(3, H, W)`` in ``[0, 1]`` (float32), and
    - a label tensor of shape ``(H, W)`` (int64).

    Labels are assumed to be Cityscapes **trainIds** (0–18; 255 = ignore).
    You can (a) keep only specific classes and set all others to ``ignore_idx``,
    or (b) additionally **remap kept classes to a compact range** ``0..K-1``.

    Args:
        root (str):
            Path to the Cityscapes root directory (containing ``leftImg8bit`` and
            ``gtFine`` or ``gtCoarse``).
        split (str, optional):
            Dataset split: ``"train"``, ``"val"``, or ``"test"``. Defaults to ``"train"``.
        mode (str, optional):
            Annotation quality: ``"fine"`` or ``"coarse"``. Defaults to ``"fine"``.
        valid_classes (Sequence[int] | None, optional):
            List of **trainId** classes to keep (e.g., ``[11, 12, 13]`` for person, rider, car).
            If ``None``, all 19 train classes are considered valid. Defaults to ``None``.
        ignore_idx (int, optional):
            Label index used for ignored pixels. Defaults to ``255``.
        remap_to_compact (bool, optional):
            If ``True``, remap kept classes to a compact range ``0..K-1`` and set all
            other pixels to ``ignore_idx``. Use this when your model head has **K** channels.
            If ``False``, keep original trainIds for the kept classes and set others to
            ``ignore_idx`` (use this with a **19-channel** head). Defaults to ``True``.
        joint_transform (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] | None, optional):
            Transform applied to **both** image and mask (e.g., resize/crop/flip). Make sure
            to use **nearest-neighbor** for masks to avoid mixing IDs. Defaults to ``None``.
        image_transform (Callable[[Tensor], Tensor] | None, optional):
            Transform applied **only** to the image after ``joint_transform`` (e.g., normalization).
            Defaults to ``None``.
        mask_transform (Callable[[Tensor], Tensor] | None, optional):
            Transform applied **only** to the mask after ``joint_transform``. Defaults to ``None``.

    Attributes:
        num_classes (int):
            Number of classes expected by the model head.
            - ``len(valid_classes)`` if ``remap_to_compact=True``.
            - ``19`` if ``remap_to_compact=False``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            ``(image, mask)`` where ``image`` is ``(3, H, W)`` float32 in ``[0, 1]``,
            and ``mask`` is ``(H, W)`` int64 with values in:
            - ``{0..K-1} ∪ {ignore_idx}`` if ``remap_to_compact=True``
            - ``{trainIds you kept} ∪ {ignore_idx}`` if ``remap_to_compact=False``

    Notes:
        - This class requests ``target_type="semantic"`` from ``Cityscapes``, which yields
          **trainId** masks. If you have **labelId** or instance masks, convert to trainIds
          before using or extend the class to add that mapping.
        - For validation/metrics, pass the same ``ignore_idx`` to your evaluator.

    Examples:
        Basic usage with compact labels (model head has K channels):

        >>> keep = [11, 12, 13]  # person, rider, car (trainIds)
        >>> ds = CityscapesSubset(
        ...     root="/path/to/cityscapes",
        ...     split="train",
        ...     mode="fine",
        ...     valid_classes=keep,
        ...     ignore_idx=255,
        ...     remap_to_compact=True,
        ... )
        >>> img, mask = ds[0]
        >>> img.shape, mask.shape
        (torch.Size([3, H, W]), torch.Size([H, W]))
        >>> mask.unique()  # doctest: +ELLIPSIS
        tensor([0, 1, 2, 255])  # kept classes mapped to 0..2, others ignored

        Keep original trainIds (model head has 19 channels):

        >>> ds = CityscapesSubset(
        ...     root="/path/to/cityscapes",
        ...     split="val",
        ...     valid_classes=[11, 12, 13],
        ...     remap_to_compact=False,
        ... )
        >>> mask = ds[0][1]
        >>> mask.unique()  # doctest: +ELLIPSIS
        tensor([ 11, 12, 13, 255])

        With a joint resize (bilinear for image, nearest for mask):

        >>> from torchvision.transforms import InterpolationMode
        >>> import torchvision.transforms.functional as TF
        >>> class JointResize:
        ...     def __init__(self, size=(512, 1024)):
        ...         self.size = size
        ...     def __call__(self, img, mask):
        ...         img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        ...         mask = TF.resize(mask.unsqueeze(0).float(), self.size,
        ...                         interpolation=InterpolationMode.NEAREST).squeeze(0).long()
        ...         return img, mask
        >>> ds = CityscapesSubset(root="/path/to/cityscapes", joint_transform=JointResize())
        >>> img, mask = ds[0]

        Dataloader + loss:

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
        >>> import torch.nn as nn
        >>> criterion = nn.CrossEntropyLoss(ignore_index=255)
        >>> images, masks = next(iter(loader))
        >>> logits = torch.randn(images.size(0), ds.num_classes, images.size(2), images.size(3))
        >>> loss = criterion(logits, masks)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: str = "semantic",
        valid_classes: Optional[Sequence[int]] = None,  
        ignore_idx: int = 255,
        remap_to_compact: bool = True,                
        joint_transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transform=None,
            target_transform=None,
        )
        self.keep_ids = list(range(19)) if valid_classes is None else list(valid_classes)
        self.class_names = [class_labels[i].name for i in self.keep_ids]
        self.ignore_idx = int(ignore_idx)
        self.remap_to_compact = bool(remap_to_compact)

        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if self.remap_to_compact:
            # Map kept trainIds -> 0..K-1; others -> ignore_idx
            lut = torch.full((256,), self.ignore_idx, dtype=torch.long)
            for new_id, old_id in enumerate(self.keep_ids):
                lut[old_id] = new_id
            self._lut_remap = lut
        else:
            allowed = torch.zeros(256, dtype=torch.bool)
            allowed[self.keep_ids] = True
            allowed[self.ignore_idx] = True
            self._allowed = allowed

    def _pil_mask_to_long(self, mask_pil) -> torch.Tensor:
        return TF.pil_to_tensor(mask_pil).squeeze(0).long()
    
    def _filter_or_remap(self, mask: torch.Tensor) -> torch.Tensor:
        if self.remap_to_compact:
            return self._lut_remap[mask]
        out = mask.clone()
        out[~self._allowed[out]] = self.ignore_idx
        return out

    def __getitem__(self, index: int):
        img_pil, mask_pil = super().__getitem__(index)  
        img = TF.pil_to_tensor(img_pil).float() / 255.0
        mask = self._pil_mask_to_long(mask_pil)

        mask = self._filter_or_remap(mask)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask



if __name__ == "__main__":
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    dataset_path = "data/cityscapes"

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    cityscapes_train = CityscapesSubset(
        root=dataset_path,
        split="train",
        mode="fine",
        valid_classes=valid_classes,
        ignore_idx=255,
        remap_to_compact=True,
        joint_transform=JointResize((512, 1024)),
        image_transform=Normalize(IMAGENET_MEAN, IMAGENET_STD, inplace=True)
    )

    print(f"Class names: {cityscapes_train.class_names}")
