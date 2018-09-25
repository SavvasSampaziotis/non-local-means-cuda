#!/bin/bash
echo "Sending .bin.in files to diades..."

scp -P 2288 -r  ../data/*.bin.in sampazio@diades.ee.auth.gr:~/nlm/

echo "DONE"