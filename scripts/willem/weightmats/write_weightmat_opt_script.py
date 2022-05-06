from subprocess import call
import os

from datarep import paths

option_str = "--nhidden 10 25 50 100 250 500 --datapath %s --path %s"%(paths.tool_path,
                                                                            os.path.join(paths.data_path, "weight_matrices/"))

with open('run_opt.sh', 'w') as file:
    # file.write("nohup python3 allopt.py --methods \'pmd\' --ndata 200000" + option_str + " &\n")
    # file.write("nohup python3 allopt.py --methods \'pmdd\' --ndata 200000 " + option_str + " &\n")
    file.write("nohup python3 allopt.py --methods \'bmd\' --ndata 10000 " + option_str + " &\n")
    file.write("nohup python3 allopt.py --methods \'bmdd\' --ndata 10000 " + option_str + " &\n")