from subprocess import call
import os

from datarep import paths

option_str = "--nhidden 10 25 50 100 250 500 --nbfactor 100 --nepoch 400 --path %s"%(os.path.join(paths.result_path, "results_bias_opt/"))

option_str2a = "--method pca ica rg rp rpw sc"
option_str2b = "--method scd sm bmd bmdd pmd pmdd"


with open('run_opt1.sh', 'w') as file:
    file.write("nohup python3 biasopt.py --ntask 50 --tasktype 1v1 %s %s &\n"%(option_str2a, option_str))

with open('run_opt2.sh', 'w') as file:
    file.write("nohup python3 biasopt.py --ntask 47 --tasktype 1vall %s %s &\n"%(option_str2a, option_str))

with open('run_opt3.sh', 'w') as file:
    file.write("nohup python3 biasopt.py --ntask 50 --tasktype 1v1 %s %s &\n"%(option_str2b, option_str))

with open('run_opt4.sh', 'w') as file:
    file.write("nohup python3 biasopt.py --ntask 47 --tasktype 1vall %s %s &\n"%(option_str2b, option_str))