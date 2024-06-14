# Learning Causal Dynamics Models in Object-Oriented Environments

This is the source code for the paper "Learning Causal Dynamics Models in Object-Oriented Environments", which was accpeted in ICML 2024.

By Zhongwei Yu.

# Installation

First, you should have installed python >= 3.9 in your device. Then, make sure you have installed the python dependencies in `requirements.txt`. 

If you want to perform experiments in StarCraftII environments (CMS and DZB), please download and install StarCraftII in your device, and set the environment variable `SC2PATH` correctly. 

## Usage

### Configure experiments

We use the class `Config` to contain hyper-parameters and basic configurations of experiments (i.e., number of samples, number of iterations, the device, etc). You can specify these hyper-parameters in `configs.py` as you want (check `config_mouse`, `config_block`, `CONFIG_CMS` and `CONFIG_DZB`).

You can specify which experiment to perform by modifying `run.py`:
* `experiment_1` trains dynamics models in a task. Then it measures the log-likelihoods, and evaluate the average episodic returns of planning. You can select the environments and approaches by modifying annotations.
* `experiment_2` trains OO models in the seen tasks of Mouse environment. Then it measures log-likelihoods and episodic returns in both seen and unseen tasks.

### Run experiments

To run experiments, simply use the following command:

```
python run.py --seed=[SEED]
```

### Check your results

All results, arguments, and logs of experiments will be stored in the `experiments` directory. If it does not exists, it will be automatically created when you run any experiment. The results will be saved at the
```
./experiments/[ENVIRONMENT]/train[APPROACH]/seed-[SEED]
```

We provide a tool in `show_results.py` to conveniently examine the results of different seeds. In the following, we show an example of how to use this tool.

```
> python show_results.py
The root directory has not been set yet. Use 'root' command to set the root.
This program shows results saved in experiments.
show-result > root experiments/cms/trainooc
Found results.json in the following runs:
- seed-5
- seed-2
- seed-4
- seed-6
- seed-9
- seed-8
- seed-3
- seed-1
- seed-7
- seed-10
show-result > set_filter seed-(6|7|8|9|10)
Found results.json in the following runs:
- seed-6
- seed-9
- seed-8
- seed-7
- seed-10
show-result > show test/loglikelihood
seed-6: 8.987736654281616
seed-9: 8.26842833518982
seed-8: 8.839284439086914
seed-7: 9.881512908935546
seed-10: 8.59796847820282
show-result > stat test/return
mean: 3.4239471440750444
max: 9.895652173913073
min: -5.208333333333319
std: 6.275761134625612
median 5.711764705882381
```
