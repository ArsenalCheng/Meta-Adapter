import random
from collections import defaultdict
import torchvision
import glob
import os
import torch
import torch.utils.data
import torchvision.models
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as Image

imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                    "box turtle", "banded gecko", "green iguana", "Carolina anole",
                    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                    "American alligator", "triceratops", "worm snake", "ring-necked snake",
                    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
                    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
                    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
imagenet_templates = ["itap of a {}.",
                      "a bad photo of the {}.",
                      "a origami {}.",
                      "a photo of the large {}.",
                      "a {} in a video game.",
                      "art of the {}.",
                      "a photo of the small {}."]
base_idx = [986, 985, 668, 430, 14, 974, 685, 607, 537, 466, 90, 24, 993, 984, 933, 927, 800, 781, 679, 645, 573, 565,
            510, 476, 340, 339, 333, 283, 95, 89, 983, 916, 820, 701, 614, 554, 458, 444, 400, 396, 323, 322, 145, 143,
            69, 13, 0, 996, 959, 895, 890, 874, 802, 779, 425, 404, 399, 388, 284, 275, 139, 137, 98, 87, 989, 982,
            964, 955, 926, 924, 922, 863, 805, 795, 755, 746, 640, 555, 535, 533, 500, 475, 351, 325, 293, 289, 255,
            148, 144, 135, 15, 9, 992, 903, 873, 867, 832, 803, 739, 688, 671, 628, 625, 580, 574, 560, 547, 496, 332,
            330, 321, 320, 292, 195, 149, 138, 76, 19, 11, 10, 980, 957, 937, 936, 917, 900, 878, 763, 687, 576, 564,
            532, 471, 410, 383, 382, 346, 336, 294, 286, 268, 181, 131, 130, 118, 88, 51, 995, 965, 963, 946, 825, 766,
            752, 719, 661, 611, 586, 546, 450, 449, 424, 407, 391, 352, 350, 324, 316, 309, 300, 291, 146, 91, 84, 82,
            80, 48, 25, 18, 12, 8, 991, 953, 886, 822, 780, 736, 732, 682, 627, 557, 528, 524, 498, 486, 477, 474, 437,
            403, 387, 367, 365, 363, 347, 344, 317, 306, 301, 259, 251, 147, 75, 16, 1, 994, 956, 938, 918, 915, 884,
            775, 734, 703, 690, 672, 563, 548, 525, 511, 454, 395, 376, 354, 338, 305, 299, 205, 178, 152, 33, 981,
            962, 958, 948, 945, 944, 934, 850, 847, 791, 783, 727, 605, 603, 568, 562, 520, 518, 467, 401, 393, 386,
            295, 258, 111, 28, 22, 4, 952, 760, 743, 695, 694, 642, 610, 597, 594, 551, 540, 531, 483, 465, 342, 308,
            296, 260, 223, 210, 150, 140, 127, 116, 105, 104, 102, 96, 70, 30, 950, 935, 932, 921, 919, 892, 880, 829,
            768, 761, 713, 712, 711, 654, 639, 621, 595, 592, 561, 448, 440, 439, 420, 406, 398, 335, 307, 279, 254,
            245, 213, 208, 132, 128, 108, 100, 31, 17, 997, 931, 888, 881, 866, 860, 853, 849, 833, 797, 741, 723, 652,
            649, 637, 515, 433, 426, 402, 397, 349, 253, 252, 243, 235, 222, 180, 174, 171, 136, 133, 109, 107, 39, 29,
            20, 951, 897, 865, 796, 759, 509, 443, 384, 355, 288, 276, 274, 247, 217, 194, 183, 141, 123, 92, 77, 68,
            45, 37, 21, 913, 912, 858, 812, 786, 758, 756, 751, 714, 709, 697, 630, 575, 522, 491, 487, 480, 431, 421,
            364, 357, 328, 261, 214, 153, 117, 74, 971, 967, 940, 907, 882, 872, 871, 862, 844, 843, 839, 827, 789,
            726, 720, 646, 613, 570, 517, 495, 453, 392, 337, 329, 297, 287, 270, 249, 230, 203, 182, 161, 156, 142,
            106, 81, 50, 973, 852, 835, 788, 717, 707, 704, 698, 643, 635, 629, 617, 577, 552, 543, 539, 468, 428, 422,
            343, 327, 298, 234, 216, 190, 71, 61, 53, 34, 966, 939, 846, 831, 747, 686, 650, 620, 553, 526, 514, 436,
            429, 366, 318, 273, 241, 209, 169, 115, 113, 72, 954, 941, 929, 901, 889, 883, 879, 877, 823, 798, 777,
            770, 757, 669, 662, 660, 657, 647, 588, 571, 521, 470, 452, 442, 358, 334, 319, 239, 228, 207, 173, 159,
            129, 125, 122, 86, 38, 869, 864, 793, 776, 769, 684, 666, 655, 634, 632, 615, 612, 608, 569, 559, 508, 484,
            432, 378, 375, 370, 362, 348, 313, 302, 263, 262, 256, 237, 199, 176, 168, 120, 57, 56, 49, 41, 7, 2, 969,
            949, 819, 806, 706, 674, 626, 616, 589, 513, 481, 462, 441, 427, 415, 379, 373, 361, 360, 244, 191, 93, 42,
            5, 3, 990, 979, 977, 925, 894, 716, 675, 624, 606, 585, 538, 435, 423, 408, 405, 385, 304, 179, 175, 165,
            164, 94, 85, 65, 961, 920, 857, 830, 826, 774, 738, 724, 692, 581, 534, 490, 485, 478, 472, 369, 311, 290,
            232, 221, 218, 212, 162, 151, 97, 43, 35, 32, 26, 23, 987, 978, 943, 887, 861, 851, 821, 815, 764, 762,
            740, 729, 699, 665, 636, 622, 601, 593, 591, 530, 507, 419, 394, 368, 233, 206, 198, 154, 112, 972, 960,
            928, 909, 898]
novel_idx = [891, 875, 854, 836, 801, 773, 631, 602, 584, 558, 541, 529, 489, 460, 451, 341, 303, 277, 271, 236, 202,
             185, 184, 160, 126, 83, 64, 63, 46, 942, 910, 904, 893, 817, 808, 794, 785, 656, 651, 599, 598, 583, 582,
             579, 572, 544, 497, 492, 417, 414, 380, 331, 281, 224, 196, 188, 103, 99, 44, 845, 834, 814, 811, 809,
             807, 790, 737, 689, 678, 641, 578, 566, 549, 527, 506, 479, 457, 456, 413, 377, 372, 315, 310, 225, 197,
             167, 124, 114, 79, 930, 856, 855, 841, 721, 590, 542, 459, 447, 390, 371, 272, 265, 220, 192, 170, 62, 27,
             975, 914, 765, 735, 710, 708, 683, 503, 502, 501, 463, 455, 314, 264, 200, 121, 119, 52, 908, 896, 870,
             799, 673, 663, 596, 494, 411, 246, 229, 211, 40, 36, 6, 911, 772, 754, 753, 728, 623, 523, 512, 280, 267,
             227, 219, 177, 172, 166, 906, 859, 840, 804, 792, 748, 696, 691, 545, 504, 464, 434, 242, 238, 186, 110,
             999, 976, 923, 745, 715, 567, 482, 473, 285, 266, 204, 187, 78, 988, 970, 968, 838, 667, 659, 644, 619,
             556, 257, 158, 157, 66, 58, 55, 848, 778, 693, 680, 604, 412, 278, 250, 134, 842, 824, 816, 733, 718, 676,
             648, 519, 438, 374, 356, 353, 312, 163, 67, 749, 742, 731, 725, 722, 653, 633, 609, 345, 226, 54, 998,
             828, 505, 101, 947, 705, 670, 536, 418, 269, 813, 488, 445, 409, 201, 155, 59, 47, 905, 771, 677, 664, 469,
             446, 381, 240, 193, 899, 885, 876, 818, 787, 782, 767, 730, 461, 73, 902, 868, 810, 618, 499, 326, 189,
             784, 700, 600, 493, 416, 248, 215, 702, 658, 550, 282, 231, 681, 587, 359, 389, 837, 750, 744, 638, 516,
             60]


class imagenet_test_dataset(Dataset):
    def __init__(self, valdir, preprocess):
        f = open(os.path.join(valdir, 'val.txt'))
        target = f.read().splitlines()
        self.target = {}
        for idx in range(len(target)):
            self.target[target[idx].split(' ')[0]] = target[idx].split(' ')[1]

        img_list = glob.glob(valdir + '/*.JPEG')

        self.img_list = []

        for idx in range(len(img_list)):
            current_name = os.path.basename(img_list[idx])
            if int(self.target[current_name]) in novel_idx:
                self.img_list.append(img_list[idx])

        self.size = len(self.img_list)

        self.transform = preprocess

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        ind = os.path.basename(self.img_list[index])
        target = int(self.target[ind])
        return img, target

    def __len__(self):
        return self.size


class ImageNet():
    dataset_dir = 'imagenet'

    def __init__(self, root, num_shots):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        self.train = torchvision.datasets.ImageFolder(os.path.join(self.image_dir, 'train'), transform=train_preprocess)
        self.full = torchvision.datasets.ImageFolder(os.path.join(self.image_dir, 'train'), transform=train_preprocess)
        self.val = torchvision.datasets.ImageFolder(os.path.join(self.image_dir, 'val'), transform=test_preprocess)

        self.template = imagenet_templates
        self.classnames = imagenet_classes

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
        imgs = []
        targets = []
        few_shot_base = defaultdict(list)
        for idx in range(len(self.classnames)):
            if idx in base_idx:
                few_shot_base[idx] = split_by_label_dict[idx]
        split_by_label_dict = few_shot_base
        for label, items in split_by_label_dict.items():
            if num_shots > 0:
                imgs = imgs + random.sample(items, num_shots)
                targets = targets + [label for _ in range(num_shots)]
            else:
                imgs = imgs + items
                targets = targets + [label for _ in range(len(items))]
        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = imgs

        val_imgs = []
        val_targets = []
        for i in range(len(self.val.imgs)):
            if self.val.targets[i] in novel_idx:
                val_imgs.append(self.val.imgs[i])
                val_targets.append(self.val.targets[i])
        self.val.imgs = val_imgs
        self.val.targets = val_targets
        self.val.samples = val_imgs

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.full.imgs)):
            split_by_label_dict[self.full.targets[i]].append(self.full.imgs[i])
        imgs_full = []
        targets_full = []
        for label, items in split_by_label_dict.items():
            if num_shots > 0:
                imgs_full = imgs_full + random.sample(items, num_shots)
                targets_full = targets_full + [label for _ in range(num_shots)]
            else:
                imgs_full = imgs_full + items
                targets_full = targets_full + [label for _ in range(len(items))]
        self.full.imgs = imgs_full
        self.full.targets = targets_full
        self.full.samples = imgs_full
