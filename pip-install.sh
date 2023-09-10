#!/bin/bash

pip install --upgrade pip
for line in $(cat requirements.txt)
do
    pip install $line
done
