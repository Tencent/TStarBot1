import os
import random


save_model_dir = './checkpoints'
log_dir = './log_hyper'
local_log = './hyper.log'
exps_num = 2
rand_patterns = {'memory_size':['enum', 100000, 50000, 50000, 20000],
                 'init_memory_size':['enum', 2000, 5000, 10000, 20000],
                 'eps_decay':['enum', 500000, 200000, 100000, 50000],
                 'learning_rate':['log-uniform', -8, -2],
                 'momentum':['enum', 0.95, 0.9, 0.0],
                 'gradient_clipping':['enum', 1.0, 1000.0, 10000000000000000.0],
                 'batch_size':['enum', 64, 128, 256],
                 'discount':['enum', 0.999, 0.99],
                 'agent':['enum', 'dqn', 'double_dqn'],
                 'target_update_freq':['enum', 500, 1000, 5000, 10000],
                 'optimize_freq':['enum', 1, 4],
                 'use_batchnorm':['bool'],
                 'allow_eval_mode':['bool']}


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
    cmds = ('CUDA_VISIBLE_DEVICES=%d python -u train_sc2_zerg_dqn.py'
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
