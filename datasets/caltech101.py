import os
from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']
base_classes = ['windsor_chair', 'trilobite', 'tick', 'sunflower', 'strawberry', 'stop_sign', 'stegosaurus', 'soccer_ball', 'rooster', 'pyramid',
                'pizza', 'panda', 'pagoda', 'okapi', 'motorbike', 'metronome', 'laptop', 'inline_skate', 'headphone', 'gramophone', 'ewer',
                'dollar_bill', 'dalmatian', 'car_side', 'cannon', 'buddha', 'brain', 'bonsai', 'barrel', 'accordion', 'airplane', 'watch',
                'starfish', 'helicopter', 'revolver', 'ferry', 'joshua_tree', 'yin_yang', 'wheelchair', 'nautilus', 'emu', 'grand_piano', 'stapler',
                'pigeon', 'menorah', 'water_lilly', 'saxophone', 'cougar_face', 'platypus', 'garfield', 'binocular', 'sea_horse', 'cup', 'kangaroo',
                'hedgehog', 'bass', 'hawksbill', 'camera', 'umbrella', 'cougar_body', 'dolphin', 'scorpion', 'minaret', 'llama', 'wrench',
                'scissors', 'butterfly', 'snoopy', 'euphonium', 'ceiling_fan']
novel_classes = ['beaver', 'leopard', 'mayfly', 'ibis', 'brontosaurus', 'elephant', 'schooner', 'flamingo_head', 'gerenuk', 'flamingo',
                 'mandolin', 'crocodile', 'chandelier', 'face', 'crayfish', 'anchor', 'rhino', 'lamp', 'lotus', 'dragonfly', 'electric_guitar',
                 'wild_cat', 'octopus', 'cellphone', 'lobster', 'ketch', 'ant', 'chair', 'crab', 'crocodile_head']


class Caltech101(DatasetBase):


    dataset_dir = 'caltech-101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        few_shot_base = []
        for item in train:
            if item.classname in base_classes:
                few_shot_base.append(item)
        few_shot_base = self.generate_fewshot_dataset(few_shot_base, num_shots=num_shots)
        few_shot_full = self.generate_fewshot_dataset(val, num_shots=num_shots)

        test_novel = []
        for item in test:
            if item.classname in novel_classes:
                test_novel.append(item)
        test_novel = self.generate_fewshot_dataset(test_novel, num_shots=num_shots)

        super().__init__(train=few_shot_base, full=few_shot_full, val=test_novel)
