from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


def print_arguments(flags_FLAGS):
  arg_name_list = dir(flags.FLAGS)
  black_set = set(['alsologtostderr',
                   'log_dir',
                   'logtostderr',
                   'showprefixforinfo',
                   'stderrthreshold',
                   'v',
                   'verbosity',
                   '?',
                   'use_cprofile_for_profiling',
                   'help',
                   'helpfull',
                   'helpshort',
                   'helpxml',
                   'profile_file',
                   'run_with_profiling',
                   'only_check_args',
                   'pdb_post_mortem',
                   'run_with_pdb'])
  print("---------------------  Configuration Arguments --------------------")
  for arg_name in arg_name_list: 
    if not arg_name.startswith('sc2_') and arg_name not in black_set:
      print("%s: %s" % (arg_name, flags_FLAGS[arg_name].value))
  print("-------------------------------------------------------------------")
