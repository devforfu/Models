#!/usr/bin/env bash

python dogsbreeds.py \
--model-name=${1:-default} \
--preprocessing=${2:-model} \
--n-epochs=100 \
--batch-size=128 \
--lr=0.001 \
--early-stopping \
--early-stopping-patience=25 \
--yes
