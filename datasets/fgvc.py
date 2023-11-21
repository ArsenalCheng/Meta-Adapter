import os

from .utils import Datum, DatasetBase

template = ['a photo of a {}, a type of aircraft.']

base_classes = ['Eurofighter Typhoon', 'Hawk T1', 'Spitfire', 'F-16A/B', 'DH-82', 'C-130', 'A380', 'F/A-18',
                'Cessna 208', 'Il-76', 'Embraer Legacy 600', 'BAE 146-200', 'ATR-72', 'Global Express', 'DC-3', 'A318',
                '777-300', 'A310', 'DC-8', 'DHC-1', 'Challenger 600', 'A340-600', 'A340-200', 'Fokker 50',
                'Falcon 2000', 'MD-11', 'Gulfstream V', 'A319', 'Fokker 70', 'DC-10', 'A330-300', 'A320', '777-200',
                'SR-20', 'DHC-6', 'Cessna 172', 'DHC-8-100', 'DC-6', 'Beechcraft 1900', '707-320', 'Cessna 560',
                'A340-300', 'DC-9-30', 'Fokker 100', 'Cessna 525', '747-300', '727-200', 'Metroliner', 'Yak-42',
                'Tu-134', 'Saab 340', 'Saab 2000', 'PA-28', 'ERJ 145', 'DHC-8-300', 'C-47', 'ATR-42', 'A330-200',
                '767-200', 'BAE 146-300', '757-200', 'Model B200', 'MD-90', 'Falcon 900', 'Dornier 328', 'A340-500',
                '747-400', '747-100', '737-400', 'MD-80']
novel_classes = ['Gulfstream IV', 'CRJ-200', 'Boeing 717', '747-200', '737-800', 'Tu-154', 'Tornado', 'MD-87', 'L-1011',
                 'ERJ 135', 'EMB-120', 'E-195', 'E-190', 'E-170', 'DR-400', 'CRJ-900', 'CRJ-700', 'BAE-125', 'An-12',
                 'A321', 'A300B4', '767-400', '767-300', '757-300', '737-900', '737-700', '737-600', '737-500',
                 '737-300', '737-200']


class FGVCAircraft(DatasetBase):
    dataset_dir = 'fgvc_aircraft'

    def __init__(self, root, num_shots):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt')
        val = self.read_data(cname2lab, 'images_variant_val.txt')
        test = self.read_data(cname2lab, 'images_variant_test.txt')

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

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)

        return items
