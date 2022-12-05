from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import mxnet as mx
from data.data_pipe import  *


def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list


def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

class train_dataset(Dataset):
    def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
        self.imgs = bcolz.carray(rootdir = imgs_bcolz)
        self.labels = bcolz.carray(rootdir = label_bcolz)
        self.h_flip = h_flip
        self.length = len(self.imgs) - 1
        if h_flip:
            self.transform = trans.Compose([
                trans.ToPILImage(),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.class_num = self.labels[-1] + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = torch.tensor(self.imgs[index+1], dtype=torch.float)
        label = torch.tensor(self.labels[index+1], dtype=torch.long)
        if self.h_flip:
            img = de_preprocess(img)
            img = self.transform(img)
        return img, label