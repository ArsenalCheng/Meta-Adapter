import os

from .utils import Datum, DatasetBase

from .oxford_pets import OxfordPets


template = ['a photo of a person doing {}.']
base_classes = ['Typing', 'Table_Tennis_Shot', 'Soccer_Penalty', 'Playing_Guitar', 'Military_Parade', 'Ice_Dancing', 'Bowling',
                'Blowing_Candles', 'Billiards', 'Bench_Press', 'Field_Hockey_Penalty', 'Baby_Crawling', 'Writing_On_Board',
                'Basketball_Dunk', 'Horse_Race', 'Sumo_Wrestling', 'Surfing', 'Clean_And_Jerk', 'Pull_Ups', 'Rock_Climbing_Indoor',
                'Playing_Violin', 'Playing_Piano', 'Apply_Eye_Makeup', 'Horse_Riding', 'Sky_Diving', 'Tai_Chi', 'Rafting', 'Playing_Dhol',
                'Breast_Stroke', 'Fencing', 'Cutting_In_Kitchen', 'Punch', 'Golf_Swing', 'Playing_Sitar', 'Band_Marching', 'Biking',
                'Mopping_Floor', 'Shaving_Beard', 'Uneven_Bars', 'Handstand_Pushups', 'Brushing_Teeth', 'Baseball_Pitch', 'Rowing',
                'Blow_Dry_Hair', 'Tennis_Swing', 'Drumming', 'Diving', 'Archery', 'Playing_Flute', 'Walking_With_Dog', 'Skate_Boarding',
                'Cliff_Diving', 'Boxing_Punching_Bag', 'Knitting', 'Cricket_Shot', 'Playing_Cello', 'Skiing', 'Playing_Tabla', 'Hula_Hoop',
                'Haircut', 'Pommel_Horse', 'Trampoline_Jumping', 'Skijet', 'Basketball', 'Salsa_Spin', 'Long_Jump', 'Apply_Lipstick',
                'Volleyball_Spiking', 'Juggling_Balls', 'Floor_Gymnastics']
novel_classes = ['High_Jump', 'Front_Crawl', 'Pole_Vault', 'Hammer_Throw', 'Pizza_Tossing', 'Swing', 'Yo_Yo', 'Shotput', 'Head_Massage',
                 'Jump_Rope', 'Soccer_Juggling', 'Hammering', 'Mixing', 'Kayaking', 'Cricket_Bowling', 'Jumping_Jack', 'Boxing_Speed_Bag',
                 'Javelin_Throw', 'Handstand_Walking', 'Lunges', 'Push_Ups', 'Throw_Discus', 'Wall_Pushups', 'Nunchucks', 'Frisbee_Catch',
                 'Body_Weight_Squats', 'Rope_Climbing', 'Parallel_Bars', 'Still_Rings', 'Playing_Daf', 'Balance_Beam']


class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        few_shot_base = []
        for item in train:
            if item.classname in base_classes:
                few_shot_base.append(item)
        few_shot_base = self.generate_fewshot_dataset(few_shot_base, num_shots=num_shots)
        few_shot_full = self.generate_fewshot_dataset(val, num_shots=16)

        test_novel = []
        for item in test:
            if item.classname in novel_classes:
                test_novel.append(item)
        test_novel = self.generate_fewshot_dataset(test_novel, num_shots=num_shots)

        super().__init__(train=few_shot_base, val=test_novel, full=few_shot_full)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')[0] # trainlist: filename, label
                action, filename = line.split('/')
                label = cname2lab[action]

                elements = re.findall('[A-Z][^A-Z]*', action)
                renamed_action = '_'.join(elements)

                filename = filename.replace('.avi', '.jpg')
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=renamed_action
                )
                items.append(item)

        return items
