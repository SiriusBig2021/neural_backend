

class NNDataGenerator():

    def __init__(self, mode, data, cfg):

        self.mode = mode
        self.data = data
        self.cfg = cfg

    def __len__(self):
        return int(len(self.data) / self.cfg["batch_size"])

    def on_epoch_end(self):
        rnd = random.random() * 10000
        random.Random(rnd).shuffle(self.data)

    def process_img(self, img):

        img = resize_image(img=img,
                           height=self.cfg["input_shape"][0],
                           width=self.cfg["input_shape"][1],
                           letterbox=False)

        if self.mode == "train":
            img = self.cfg["aug_seq"].augment_images([img])[0]

        return img / 255.

    def generate_batch(self, indexes):

        images_batch = []
        labels_batch = []

        for i in indexes:

            sample = self.data[i]

            img = cv2.imread(sample["img"])
            img = self.process_img(img)
            images_batch.append(img)

            label = sample["label"]
            labels_batch.append(label)

            if img is None:
                print("None image:", sample["img"])

        images_batch = np.array(images_batch)
        labels_batch = [np.array(x) for x in labels_batch]

        return images_batch, labels_batch

    def __getitem__(self, item):
        return self.generate_batch([i + item * self.cfg["batch_size"] for i in range(self.cfg["batch_size"])])

    def show_data(self):

        for i in range(self.__len__()):
            batch = self.__getitem__(i)

            for n, img in enumerate(batch[0]):
                print(self.cfg["classes"])
                for br in batch[1]:
                    print(br[n])
                show_image(img, win_name=self.name)
                print()