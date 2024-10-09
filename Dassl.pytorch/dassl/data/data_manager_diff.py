import torch
import os
import json
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform


def get_dataset_info(cfg):
    if cfg.DATASET.NAME == "DescribableTextures":
        dataset_dir = os.path.join(cfg.DATASET.ROOT, 'dtd')
        image_dir = os.path.join(dataset_dir, "images")
        return dataset_dir, image_dir
    elif cfg.DATASET.NAME == "FGVCAircraft":
        dataset_dir = os.path.join(cfg.DATASET.ROOT, 'fgvc_aircraft')
        image_dir = os.path.join(dataset_dir, "images")
        return dataset_dir, image_dir
    elif cfg.DATASET.NAME == "OxfordFlowers":
        dataset_dir = os.path.join(cfg.DATASET.ROOT, 'oxford_flowers')
        image_dir = os.path.join(dataset_dir, "jpg")
        return dataset_dir, image_dir
    else:
        print(f"Clone image not found for dataset {cfg.DATASET.NAME}")
        return None

def get_clone_image_path(dataset_dir, image_dir, item):
    clone_dir = os.path.join(dataset_dir, 'jpg_cloned')
    relative_path = os.path.relpath(item.impath, image_dir)
    clone_img = os.path.join(clone_dir, relative_path)
    if os.path.exists(clone_img):
        return clone_img
    else:
        print(f"Clone image not found for item {item.impath}")
        return None

def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)
        info = get_dataset_info(cfg)
        if info is not None:
            self.dataset_dir, self.image_dir = info
            self.clone_image_dir = os.path.join(self.dataset_dir, 'jpg_cloned') # clone image directory
            captions_file = os.path.join(self.dataset_dir, "image_captions_with_classes_v1.json")
            if os.path.exists(self.clone_image_dir):
                # read the captions file
                with open(captions_file, 'r') as f:
                    self.json_captions = json.load(f)
        else:
            self.dataset_dir, self.image_dir = None, None
            print(f"Clone image not found for dataset {cfg.DATASET.NAME}")

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "classname": item.classname,
            "index": idx
        }

        # Get clone image path
        if self.dataset_dir is not None:
            relative_path = os.path.relpath(item.impath, self.image_dir)
            clone_impath = os.path.join(self.clone_image_dir, relative_path)
            # clone_impath = get_clone_image_path(self.dataset_dir, self.image_dir, item)
            if os.path.exists(clone_impath):
                clone_img = read_image(clone_impath)
                # check clone_img is not None
                if clone_img is None:
                    print(f"Clone image not found for item {item.impath}")
                else:
                    # output["clone_img"] = self.to_tensor(clone_img)
                    if self.transform is not None:
                        if isinstance(self.transform, (list, tuple)):
                            for i, tfm in enumerate(self.transform):
                                img = self._transform_image(tfm, clone_img)
                                keyname = "clone_img"
                                if (i + 1) > 1:
                                    keyname += str(i + 1)
                                output[keyname] = img
                        else:
                            img = self._transform_image(self.transform, clone_img)
                            output["clone_img"] = img
                    else:
                        output["clone_img"] = clone_img

            # Add caption if it exists
            
            output['caption'] = self.json_captions.get(relative_path, "")
            # print(f"Caption found for item {idx}: {output['caption']}")
            # print(f"Clone image found for item {idx}: {clone_impath}")
            # print(f"Clone image found for item {idx}: {clone_impath1}")

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
