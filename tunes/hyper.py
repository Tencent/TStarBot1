import os
import random


save_model_dir = './checkpoints'
log_dir = './log_hyper'
local_log = './hyper.log'
exps_num = 8
rand_patterns = {'eps_end':['enum', 0.1, 0.1, 0.2],
                 'eps_decay':['enum', 2000000, 1000000, 1000000, 500000],
                 'learning_rate':['log-uniform', -9, -5],
                 'momentum':['enum', 0.95, 0.9],
                 'batch_size':['enum', 128, 128, 256],
                 'discount':['enum', 0.99, 0.99, 0.999],
                 'agent':['enum', 'fast_dqn', 'fast_double_dqn', 'fast_double_dqn'],
                 'target_update_freq':['enum', 2500, 5000, 5000, 10000],
                 'frame_step_ratio':['enum', 0.25, 0.5, 1.0, 2.0, 4.0]}


def gen_random_hypers(rand_patterns):
    conf = ""
    for param_name, pattern in rand_patterns.items():
        if pattern[0] == 'uniform':
            assert len(pattern) == 3, "Type 'uniform' requires 2 arguments"
            value = random.uniform(pattern[1], pattern[2])
            conf += " --%s %g" % (param_name, value)
        elif pattern[0] == 'log-uniform':
            assert len(pattern) == 3, "Type 'log-uniform' requires 2 arguments."
            value = pow(10.0, random.uniform(pattern[1], pattern[2]))
            conf += " --%s %g" % (param_name, value)
        elif pattern[0] == 'enum':
            value = random.choice(pattern[1:])
            conf += " --%s %s" % (param_name, value)
        elif pattern[0] == 'bool':
            value = random.choice([0, 1])
            if value == 1:
                conf += " --%s" % param_name
            else:
                conf += " --no%s" % param_name
        else:
            assert False, "Type %s not supported." % pattern[0]
    return conf


def hyper_tune(exp_id):
    conf = gen_random_hypers(rand_patterns)
    conf += ' --save_model_dir %s' % os.path.join(save_model_dir,
                                                  'checkpoints_%d' % exp_id)
    log_path = os.path.join(log_dir, 'log_%d' % exp_id)
    cmds = ('CUDA_VISIBLE_DEVICES=%d python -u train_sc2_zerg_dqn_v1.py'
            '%s > %s 2>&1 &' % (exp_id, conf, log_path))
    print(cmds)
    os.system(cmds)
    return cmds


if __name__ == '__main__':
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    with open(local_log, 'wt') as f:
        for i in range(exps_num):
            cmds = hyper_tune(i)
            f.write("%s\n" % cmds)
