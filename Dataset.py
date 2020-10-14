import tensorflow as tf
from IMLib.utils import *

ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
          'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
          'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
          'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
          'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
          'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
          'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
          'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
          'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
          'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
          'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
          'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
          'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

class CelebA(object):

    def __init__(self, config):
        super(CelebA, self).__init__()

        self.data_dir = config.data_dir
        self.label_dir = config.label_dir
        self.dataset_name = 'CelebA'
        self.height, self.width= config.img_size, config.img_size
        self.channel = config.output_nc
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads
        self.chosen_att_names = config.chosen_att_names

        self.img_names = np.genfromtxt(self.label_dir, dtype=str, usecols=0)
        self.img_paths = np.array([os.path.join(self.data_dir, img_name) for img_name in self.img_names[2:]])
        self.labels = self.read_txt(self.label_dir) #np.genfromtxt(self.label_dir, dtype=str, usecols=range(0, 41), delimiter='[/\s:]+')

        self.labels = self.labels[:, np.array([ATT_ID[att_name] for att_name in self.chosen_att_names])]

        self.labels = np.stack([[self.labeltoCat(item) for item in self.labels]], axis=-1)

        assert len(self.labels) == len(self.img_paths)

        self.train_images_list = self.img_paths[0:29000, ...]
        self.test_images_list = self.img_paths[29000:-1, ...]
        self.train_label = self.labels[0:29000, ...]
        self.test_label = self.labels[29000:-1, ...]

        print(self.train_images_list[0], len(self.test_images_list), len(self.train_label), len(self.test_label))

    def read_images(self, input_queue):

        content = tf.read_file(input_queue)
        img = tf.image.decode_jpeg(content, channels=self.channel)
        img = tf.cast(img, tf.float32)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.central_crop(img, central_fraction=0.9)
        img = tf.image.resize_images(img, (self.height, self.width))

        return img / 127.5 - 1.0

    def labeltoCat(self, l):
        cat = 0
        for i, item in enumerate(l):
            cat += item * pow(2, i)
        return cat

    def read_txt(self, txt_path):

        p = open(txt_path, 'r')
        lines = p.readlines()
        labels = []
        for i, line in enumerate(lines):
            if i == 0 or i == 1:
                continue
            line = line.replace('\n', '')
            list = line.split()
            label = [(int(item) + 1)/2 for item in list[1:]]
            labels.append(label)

        return np.array(labels)

    def input(self):

        train_images = tf.convert_to_tensor(self.train_images_list, dtype=tf.string)

        train_queue = tf.train.slice_input_producer([train_images], shuffle=True)
        train_images_queue = self.read_images(input_queue=train_queue[0])

        test_images = tf.convert_to_tensor(self.test_images_list, dtype=tf.string)
        test_queue = tf.train.slice_input_producer([test_images], shuffle=False)
        test_images_queue = self.read_images(input_queue=test_queue[0])

        batch_image = tf.train.shuffle_batch([train_images_queue],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=200)

        test_batch_image = tf.train.batch([test_images_queue],
                                                batch_size=self.batch_size,
                                                capacity=100,
                                                num_threads=1)

        return batch_image, test_batch_image
