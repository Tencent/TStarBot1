from absl import flags


def print_arguments(flags_FLAGS):
    arg_name_list = dir(flags.FLAGS)
    arg_name_list.remove('alsologtostderr')
    arg_name_list.remove('log_dir')
    arg_name_list.remove('logtostderr')
    arg_name_list.remove('showprefixforinfo')
    arg_name_list.remove('stderrthreshold')
    arg_name_list.remove('v')
    arg_name_list.remove('verbosity')
    arg_name_list.remove('?')
    arg_name_list.remove('use_cprofile_for_profiling')
    arg_name_list.remove('help')
    arg_name_list.remove('helpfull')
    arg_name_list.remove('helpshort')
    arg_name_list.remove('helpxml')
    arg_name_list.remove('profile_file')
    arg_name_list.remove('run_with_profiling')
    arg_name_list.remove('only_check_args')
    arg_name_list.remove('pdb_post_mortem')
    arg_name_list.remove('run_with_pdb')

    print("---------------------  Configuration Arguments --------------------")
    for arg_name in arg_name_list: 
        print("%s: %s" % (arg_name, flags_FLAGS[arg_name].value))
    print("-------------------------------------------------------------------")
