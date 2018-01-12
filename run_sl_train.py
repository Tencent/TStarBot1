import os
import traceback
from absl import app
from absl import flags

from agents.sl_agent import SLAgent
from agents.dataset import SCReplayDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("train_data_dir", None, "Directory for training data.")
flags.DEFINE_string("dev_data_dir", None, "Directory for validation data.")
flags.DEFINE_integer("num_dataloader_worker", 16, "Processes # for dataloader.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("print_freq", 50, "Frequency to print train loss.")
flags.DEFINE_integer("max_train_epochs", 10000, "Maximal training epochs.")
flags.DEFINE_enum("optimizer_type", 'adam', ['adam', 'rmsprop'], "Optimizer.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for RMSProp.")
flags.DEFINE_boolean("use_gpu", True, "Use gpu or not.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_model_freq", 1000, "Frequency to save model.")
flags.DEFINE_integer("max_sampled_dev_ins", 3200, "Sampled dev instances.")
flags.DEFINE_string("observation_filter", "", "Observation field to ignore.")
flags.mark_flag_as_required("train_data_dir")
flags.mark_flag_as_required("dev_data_dir")


def train():
    if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
        os.mkdir(FLAGS.save_model_dir)

    dataset_train = SCReplayDataset(
        FLAGS.train_data_dir, resolution=64,
        observation_filter=FLAGS.observation_filter.split(","))
    dataset_dev = SCReplayDataset(
        FLAGS.train_data_dir, resolution=64,
        observation_filter=FLAGS.observation_filter.split(","))

    agent = SLAgent(
        observation_spec=dataset_train.observation_spec,
        action_spec=dataset_train.action_spec,
        use_gpu=FLAGS.use_gpu,
        init_model_path=FLAGS.init_model_path)

    try:
        agent.train(dataset_train=dataset_train,
                    dataset_dev=dataset_dev,
                    optimizer_type=FLAGS.optimizer_type,
                    learning_rate=FLAGS.learning_rate,
                    batch_size=FLAGS.batch_size,
                    num_dataloader_worker=FLAGS.num_dataloader_worker,
                    save_model_dir=FLAGS.save_model_dir,
                    save_model_freq=FLAGS.save_model_freq,
                    print_freq=FLAGS.print_freq,
                    max_sampled_dev_ins=FLAGS.max_sampled_dev_ins,
                    max_epochs=FLAGS.max_train_epochs)
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()


def main(argv):
    train()


if __name__ == '__main__':
    app.run(main)
