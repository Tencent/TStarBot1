import os
import traceback
from absl import app
from absl import flags

from agents.sl_agent import SLAgent
from agents.dataset import SCReplayDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("train_filelist", None, "Training filelist.")
flags.DEFINE_string("dev_filelist", None, "Validation filelist.")
flags.DEFINE_integer("num_dataloader_worker", 16, "Processes # for dataloader.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("print_freq", 100, "Frequency to print train loss.")
flags.DEFINE_integer("max_train_epochs", 10000, "Maximal training epochs.")
flags.DEFINE_enum("optimizer_type", 'adam', ['adam', 'rmsprop'], "Optimizer.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for RMSProp.")
flags.DEFINE_boolean("use_gpu", True, "Use gpu or not.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_model_freq", 10000, "Frequency to save model.")
flags.DEFINE_integer("max_sampled_dev_ins", 30000, "Sampled dev instances.")
flags.DEFINE_string("observation_filter", "", "Observation field to ignore.")
flags.mark_flag_as_required("train_filelist")
flags.mark_flag_as_required("dev_filelist")

unittype_whitelist=[0, 5, 6, 11, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                    34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                    54, 55, 56, 57, 58, 130, 132, 134, 268, 341,
                    342, 472, 483, 484, 498, 500, 638, 641, 689, 691,
                    692, 734, 830]

def train():
    if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
        os.mkdir(FLAGS.save_model_dir)

    dataset_train = SCReplayDataset(
        FLAGS.train_filelist,
        resolution=64,
        unittype_whitelist=unittype_whitelist,
        observation_filter=FLAGS.observation_filter.split(","))
    dataset_dev = SCReplayDataset(
        FLAGS.train_filelist,
        resolution=64,
        unittype_whitelist=unittype_whitelist,
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
