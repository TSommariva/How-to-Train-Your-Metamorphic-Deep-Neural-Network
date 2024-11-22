#!/bin/bash
srun -Q --immediate=10 -w ailb-login-03 --partition=all_serial --account=cvcs_2023_group25 --gres=gpu:1 --time 4:00:00 --pty bash
