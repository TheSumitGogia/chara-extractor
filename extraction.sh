#!/bin/bash

range=(0 1 2)
outfolders=(features1 features2 features3)
nparams=('[15, 20, 20, 10, 10]' '[30, 40, 40, 20, 20]' '[45, 60, 60, 30, 30]') 
cnparams=(10 20 30)
for ind in "${range[@]}"; do
   python feature_parser.py -f raw_nlp -rf tokens -o "${outfolders[ind]}" -dr -n "${nparams[ind]}" -cn "${cnparams[ind]}";
done
