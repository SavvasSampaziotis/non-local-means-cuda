#!/bin/bash

echo "Making demo_nlm_cuda"
make all

echo "Sending executable to diades..."
scp -P 2288 -r ./*.out  sampazio@diades.ee.auth.gr:~/nlm

echo "DONE"
