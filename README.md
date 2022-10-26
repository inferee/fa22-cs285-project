# fa22-cs285-project
## Setup

If you choose to run locally, you will need to install some Python packages with `pip install -r requirements.txt`.

Make sure to also run `pip install -e .` in this folder.

```
python cs285/scripts/run_sac.py \
--env_name Walker2d-v4 --env_tasks forward backward jump --multitask_setting none --ep_len 1000 \
--discount 0.99 --scalar_log_freq 1000 \
-n 100000 -l 2 -s 256 -b 1000 -eb 2000 \
-lr 0.0003 --init_temperature 0.1 --exp_name walker_fbj_none \
--seed 1

python cs285/scripts/run_sac.py \
--env_name Walker2d-v4 --env_tasks forward backward jump --multitask_setting all --ep_len 1000 \
--discount 0.99 --scalar_log_freq 1000 \
-n 100000 -l 2 -s 256 -b 1000 -eb 2000 \
-lr 0.0003 --init_temperature 0.1 --exp_name walker_fbj_all \
--seed 1

python cs285/scripts/run_sac.py \
--env_name Walker2d-v4 --env_tasks forward backward --multitask_setting all --ep_len 1000 \
--discount 0.99 --scalar_log_freq 1000 \
-n 100000 -l 2 -s 256 -b 1000 -eb 2000 \
-lr 0.0003 --init_temperature 0.1 --exp_name walker_fb_all \
--seed 1

```