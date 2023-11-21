import os

from .utils import Datum, DatasetBase

from .oxford_pets import OxfordPets

template = ['a photo of a {}.']
base_classes = ['indoor florist_shop', 'skatepark', 'raft', 'oilrig', 'ball_pit', 'martial_arts_gym', 'courtroom',
                'cockpit', 'airplane_cabin', 'volcano', 'sauna', 'music_studio', 'indoor volleyball_court',
                'batters_box', 'wind_farm', 'wave', 'rock_arch', 'raceway', 'outdoor track', 'oast_house',
                'limousine_interior', 'indoor cloister', 'cemetery', 'carrousel', 'baseball stadium',
                'auto_factory', 'vineyard', 'toll_plaza', 'television_studio', 'outdoor tennis_court',
                'outdoor oil_refinery', 'manufactured_home', 'lift_bridge', 'indoor pilothouse', 'forest_road',
                'exterior covered_bridge', 'coral_reef underwater', 'bowling_alley', 'bamboo_forest', 'aquarium',
                'veterinarians_office', 'vegetation desert', 'outdoor hangar', 'dining_car', 'control_room',
                'barrel_storage wine_cellar', 'squash_court', 'sky', 'promenade_deck', 'playground',
                'platform train_station', 'pantry', 'outdoor lido_deck', 'outdoor ice_skating_rink',
                'outdoor control_tower', 'kindergarden_classroom', 'kasbah', 'islet', 'indoor brewery', 'igloo',
                'heliport', 'courthouse', 'rope_bridge', 'rice_paddy', 'racecourse', 'pulpit', 'landing_deck',
                'indoor gymnasium', 'indoor cavern', 'indoor casino', 'ice_floe', 'crevasse', 'butte', 'bus_interior',
                'boxing_ring', 'topiary_garden', 'ski_resort', 'pharmacy', 'outdoor greenhouse',
                'outdoor athletic_field', 'orchard', 'lighthouse', 'indoor wrestling_ring', 'indoor tennis_court',
                'indoor swimming_pool', 'fire_station', 'closet', 'bottle_storage wine_cellar', 'boardwalk',
                'outdoor labyrinth', 'landfill', 'indoor jail', 'iceberg', 'bullring', 'art_gallery',
                'anechoic_chamber', 'amusement_park', 'videostore', 'throne_room', 'slum', 'sandbox',
                'picnic_area', 'outdoor tent', 'laundromat', 'indoor warehouse', 'indoor ice_skating_rink',
                'hot_spring', 'exterior gazebo', 'dam', 'campus', 'aqueduct', 'windmill',
                'water_tower', 'subway_interior', 'phone_booth', 'pagoda', 'indoor escalator', 'indoor badminton_court',
                'establishment poolroom', 'discotheque', 'childs_room', 'archive', 'amphitheater', 'shop bakery',
                'riding_arena', 'residential_neighborhood', 'outdoor volleyball_court', 'outdoor general_store',
                'outdoor basketball_court', 'interior elevator', 'indoor synagogue', 'indoor firing_range',
                'gas_station', 'electrical_substation', 'driveway', 'classroom', 'basilica', 'schoolhouse',
                'physics_laboratory', 'outdoor podium', 'mausoleum', 'fountain', 'excavation', 'dorm_room',
                'cheese_factory', 'viaduct', 'utility_room', 'outdoor outhouse', 'outdoor driving_range',
                'outdoor doorway', 'music_store', 'marsh', 'locker_room', 'kitchenette', 'kitchen',
                'indoor shopping_mall', 'indoor booth', 'canyon', 'badlands', 'south_asia temple', 'shoe_shop',
                'sandbar', 'sand desert', 'restaurant_kitchen', 'outdoor bazaar', 'indoor market', 'conference_room',
                'butchers_shop', 'banquet_hall', 'vegetable_garden', 'railroad_track', 'patio', 'outdoor hot_tub',
                'medina', 'hospital_room', 'harbor', 'frontseat car_interior', 'creek', 'chalet', 'campsite',
                'boathouse', 'biology_laboratory', 'barn', 'tree_farm', 'snowfield', 'outdoor observatory',
                'indoor parking_garage', 'indoor bow_window', 'fishpond', 'elevator_shaft', 'cafeteria',
                'broadleaf forest', 'beach', 'train_railway', 'server_room', 'pasture', 'outdoor market',
                'indoor hangar', 'golf_course', 'food_court', 'corridor', 'bedroom', 'valley', 'urban canal',
                'restaurant_patio', 'public atrium', 'outdoor nuclear_power_plant', 'office cubicle', 'indoor pub',
                'highway', 'engine_room', 'dining_room', 'crosswalk', 'computer_room', 'tree_house', 'rainforest',
                'outdoor bow_window', 'outdoor apartment_building', 'lecture_room', 'indoor stage', 'indoor library',
                'indoor jacuzzi', 'indoor chicken_coop', 'indoor bazaar', 'hospital', 'hayfield', 'football stadium',
                'beauty_salon', 'skyscraper', 'putting_green', 'operating_room', 'indoor bistro', 'garbage_dump',
                'formal_garden', 'dock', 'corn_field', 'construction_site', 'ballroom', 'baggage_claim', 'art_studio',
                'wheat_field', 'sushi_bar', 'supermarket', 'ski_lodge', 'runway', 'park', 'outdoor kennel',
                'outdoor diner', 'lobby', 'indoor general_store', 'exterior balcony', 'watering_hole', 'van_interior',
                'plaza', 'outdoor arrival_gate', 'fire_escape', 'fairway', 'water moat', 'village', 'street', 'shower',
                'outdoor planetarium', 'outdoor church', 'jail_cell', 'indoor church', 'indoor cathedral',
                'candy_store', 'ticket_booth', 'staircase', 'outdoor power_plant', 'office_building', 'indoor garage',
                'catacomb', 'amusement_arcade', 'plunge waterfall', 'jewelry_shop', 'forest_path']
novel_classes = ['east_asia temple', 'dentists_office', 'castle', 'bookstore', 'arch', 'alley', 'toyshop', 'pond',
                 'platform subway_station',
                 'palace', 'outdoor chicken_coop', 'motel', 'ice_cream_parlor', 'home_office', 'clothing_store',
                 'auditorium', 'wet_bar',
                 'tower', 'swamp', 'shopfront', 'parlor', 'outdoor swimming_pool', 'outdoor mosque',
                 'outdoor cathedral', 'mountain_snowy',
                 'indoor diner', 'fastfood_restaurant', 'cultivated field', 'parking_lot', 'natural lake',
                 'herb_garden', 'basement',
                 'sea_cliff', 'indoor kennel', 'home poolroom', 'game_room', 'fan waterfall', 'conference_center',
                 'coast', 'bathroom',
                 'barndoor', 'office', 'indoor factory', 'ice_shelf', 'delicatessen', 'courtyard', 'bridge', 'abbey',
                 'veranda', 'ski_slope',
                 'shed', 'indoor mosque', 'indoor greenhouse', 'gift_shop', 'cottage_garden', 'playroom',
                 'outdoor monastery', 'indoor museum',
                 'outdoor cabin', 'indoor apse', 'hill', 'burial_chamber', 'berth', 'bar', 'airport_terminal', 'yard',
                 'stable', 'recreation_room',
                 'outdoor parking_garage', 'corral', 'thriftshop', 'natural canal', 'indoor movie_theater', 'house',
                 'attic', 'trench', 'ruin',
                 'outdoor hunting_lodge', 'interior balcony', 'home dinette', 'building_facade', 'boat_deck', 'river',
                 'ocean', 'hotel_room',
                 'baseball_field', 'cliff', 'botanical_garden', 'waiting_room', 'mountain', 'lock_chamber',
                 'indoor podium', 'door elevator', 'coffee_shop', 'bayou', 'chemistry_lab', 'assembly_line',
                 'youth_hostel', 'pavilion', 'industrial_area', 'galley',
                 'art_school', 'reception', 'outdoor hotel', 'living_room', 'wild field', 'outdoor inn',
                 'outdoor synagogue', 'indoor_procenium theater', 'restaurant', 'nursery', 'needleleaf forest',
                 'mansion', 'indoor_seats theater', 'drugstore', 'block waterfall', 'vehicle dinette',
                 'outdoor library', 'clean_room', 'backseat car_interior'
                 ]

class SUN397(DatasetBase):
    dataset_dir = 'sun397'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_SUN397.json')

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

        super().__init__(train=few_shot_base, val=test_novel, full=few_shot_full)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split('/')[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = ' '.join(names)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)

        return items
