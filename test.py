import time
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import load_dataset, visualize
from pix2pix_gan import UnetGenerator

def test(dataset, batch_size, filters, context):
    datasets = {
        "facades": True,
        "cityscapes": False,
        "maps": False
    }
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    validating_set = load_dataset(dataset, "val", batch_size, is_reversed=datasets[dataset])

    net_g = UnetGenerator(3, filters)
    net_g.load_parameters("model/{}.generator.params".format(dataset), ctx=context)

    print("Testing...", flush=True)
    for batch in validating_set:
        real_in = batch.data[0].as_in_context(context)
        real_out = batch.data[1].as_in_context(context)
        fake_out = net_g(real_in)

        for i in range(batch_size):
            plt.subplot(3, batch_size, i + 1)
            visualize(real_in[i])
            plt.subplot(3, batch_size, i + batch_size + 1)
            visualize(real_out[i])
            plt.subplot(3, batch_size, i + batch_size * 2 + 1)
            visualize(fake_out[i])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a pix2pix_gan tester.")
    parser.add_argument("--dataset", help="set the dataset used by the tester (default: facades)", type=str, default="facades")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(
        dataset = args.dataset,
        batch_size = 4,
        filters = 64,
        context = context
    )

