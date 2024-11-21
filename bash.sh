#!/bin/bash
srun -Q --immediate=10 -w ailb-login-03 --partition=all_serial --account=tesi_tsommariva --gres=gpu:1 --time 4:00:00 --pty bash
