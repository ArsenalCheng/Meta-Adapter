import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets

template = ['a centered satellite photo of {}.']

NEW_CNAMES = {
    'Annual Crop Land': 'AnnualCrop',
    'Forest': 'Forest',
    'Herbaceous Vegetation Land': 'HerbaceousVegetation',
    'Highway or Road': 'Highway',
    'Industrial Buildings': 'Industrial',
    'Pasture Land': 'Pasture',
    'Permanent Crop Land': 'PermanentCrop',
    'Residential Buildings': 'Residential',
    'River': 'River',
    'Sea or Lake': 'SeaLake'
}
base_classes = ['Forest', 'Industrial Buildings', 'Highway or Road', 'Residential Buildings', 'Pasture Land',
                'Permanent Crop Land', 'Sea or Lake']
novel_classes = ['River', 'Herbaceous Vegetation Land', 'Annual Crop Land']


class EuroSAT(DatasetBase):
    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        few_shot_base = []
        for item in train:
            if item.classname in base_classes:
                few_shot_base.append(item)
        few_shot_base = self.generate_fewshot_dataset(train, num_shots=num_shots)
        few_shot_full = self.generate_fewshot_dataset(val, num_shots=num_shots)

        test_novel = []
        for item in test:
            if item.classname in novel_classes:
                test_novel.append(item)
        test_novel = self.generate_fewshot_dataset(test_novel, num_shots=num_shots)

        self.update_classname(few_shot_base)
        self.update_classname(test_novel)
        self.update_classname(few_shot_full)
        self.update_classname(val)

        super().__init__(train=few_shot_base, full=few_shot_full, val=test_novel)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CNAMES[cname_old]
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
