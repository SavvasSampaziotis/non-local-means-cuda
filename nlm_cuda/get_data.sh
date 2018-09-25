#!/bin/bash
echo "Getting *.bin.out files from diades..."

scp -P 2288 -r sampazio@diades.ee.auth.gr:~/nlm/*.bin.out ../data/

echo "DONE"