import os
import time
import argparse
import mxnet as mx
from dataset import load_dataset
from pix2pix_gan import UnetGenerator, Discriminator, WassersteinLoss

def train(dataset, max_epochs, learning_rate, batch_size, filters, lmda, context):
    datasets = {
        "facades": True,
        "cityscapes": False,
        "maps": False
    }
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set = load_dataset(dataset, "train", batch_size, is_reversed=datasets[dataset])

    net_g = UnetGenerator(3, filters)
    net_d = Discriminator(filters)
    wgan_loss = WassersteinLoss()
    l1_loss = mx.gluon.loss.L1Loss()

    gen_params_file = "model/{}.generator.params".format(dataset)
    dis_params_file = "model/{}.discriminator.params".format(dataset)
    gen_state_file = "model/{}.generator.state".format(dataset)
    dis_state_file = "model/{}.discriminator.state".format(dataset)

    if os.path.isfile(gen_params_file):
        net_g.load_parameters(gen_params_file, ctx=context)
    else:
        net_g.initialize(mx.init.Xavier(), ctx=context)

    if os.path.isfile(dis_params_file):
        net_d.load_parameters(dis_params_file, ctx=context)
    else:
        net_d.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate, flush=True)
    trainer_g = mx.gluon.Trainer(net_g.collect_params(), "RMSProp", {
        "learning_rate": learning_rate,
    })
    trainer_d = mx.gluon.Trainer(net_d.collect_params(), "RMSProp", {
        "learning_rate": learning_rate,
        "clip_weights": 0.01
    })

    if os.path.isfile(gen_state_file):
        trainer_g.load_states(gen_state_file)

    if os.path.isfile(dis_state_file):
        trainer_d.load_states(dis_state_file)

    print("Training...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        training_L = 0.0
        training_batch = 0
        training_set.reset()

        for batch in training_set:
            training_batch += 1

            real_in = batch.data[0].as_in_context(context)
            real_out = batch.data[1].as_in_context(context)

            with mx.autograd.record():
                real_y = net_d(mx.nd.concat(real_in, real_out, dim=1))
                fake_out = net_g(real_in)
                fake_y = net_d(mx.nd.concat(real_in, fake_out.detach()))
                L = wgan_loss(fake_y, real_y)
                L.backward()
            trainer_d.step(batch_size)
            batch_L = mx.nd.mean(L).asscalar()
            if batch_L != batch_L:
                raise ValueError()

            with mx.autograd.record():
                y = net_d(mx.nd.concat(real_in, fake_out, dim=1))
                L = wgan_loss(y) + l1_loss(fake_out, real_out) * lmda
                L.backward()
            trainer_g.step(batch_size)
            gen_L = mx.nd.mean(L).asscalar()
            if gen_L != gen_L:
                raise ValueError()
                
            training_L += batch_L
            print("[Epoch %d  Batch %d]  batch_loss %.10f  gen_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                epoch, training_batch, batch_L, gen_L, training_L / training_batch, time.time() - ts
            ), flush=True)

        avg_L = training_L / training_batch
        print("[Epoch %d]  training_loss %.10f  duration %.2fs" % (
            epoch + 1, avg_L, time.time() - ts
        ), flush=True)

        net_g.save_parameters(gen_params_file)
        net_d.save_parameters(dis_params_file)
        trainer_g.save_states(gen_state_file)
        trainer_d.save_states(dis_state_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a pix2pix_gan trainer.")
    parser.add_argument("--dataset", help="set the dataset used by the trainer (default: facades)", type=str, default="facades")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.00005)", type=float, default=0.00005)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    while True:
        try:
            train(
                dataset = args.dataset,
                max_epochs = args.max_epochs,
                learning_rate = args.learning_rate,
                batch_size = 64,
                filters = 64,
                lmda = 100,
                context = context
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
