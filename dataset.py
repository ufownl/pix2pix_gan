import os
import tarfile
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def cook_image(img, image_size, is_reversed):
    h, w, _ = img.shape
    img_l = mx.image.fixed_crop(img, 0, 0, w // 2, h, size=image_size)
    img_r = mx.image.fixed_crop(img, w // 2, 0, w // 2, h, size=image_size)
    if is_reversed:
        return (
            img_r.astype("float32") / 127.5 - 1.0,
            img_l.astype("float32") / 127.5 - 1.0
        )
    else:
        return (
            img_l.astype("float32") / 127.5 - 1.0,
            img_r.astype("float32") / 127.5 - 1.0
        )

def load_dataset(name, category, batch_size, image_size=(256, 256), is_reversed=False):
    url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % (name)
    data_path = "data"
    if not os.path.exists(os.path.join(data_path, name)):
        data_file = mx.gluon.utils.download(url)
        with tarfile.open(data_file) as tar:
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=data_path)
    img_path = [os.path.join(path, f) for path, _, files in os.walk(os.path.join(data_path, name, category)) for f in files]
    imgs = [cook_image(load_image(img), image_size, is_reversed) for img in img_path]
    imgs_l = [img.T.expand_dims(0) for img, _ in imgs]
    imgs_r = [img.T.expand_dims(0) for _, img in imgs]
    return mx.io.NDArrayIter([mx.nd.concat(*imgs_l, dim=0), mx.nd.concat(*imgs_r, dim=0)], batch_size=batch_size, shuffle=True)

def visualize(img):
   plt.imshow(((img.T + 1.0) * 127.5).asnumpy().astype(np.uint8))
   plt.axis("off")


if __name__ == "__main__":
    batch_size = 4
    dataset = load_dataset("facades", "train", batch_size, is_reversed=True)
    batch = next(dataset).data
    print("batch preview: ", batch)
    for i in range(batch_size):
        plt.subplot(2, batch_size, i + 1)
        visualize(batch[0][i])
        plt.subplot(2, batch_size, i + batch_size + 1)
        visualize(batch[1][i])
    plt.show()
