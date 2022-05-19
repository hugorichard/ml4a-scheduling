#!/usr/bin/env bash

python plot_etc_vs_opt-2-jobs.py
python plot_etc_vs_opt-k-jobs-uniform.py
python plot_etc_vs_opt-vary-nsamples.py

for i in ../figures/*.pdf; do pdfcrop $i ../../6256d53e91de2395bf0d9fd6/figures/$i; done
