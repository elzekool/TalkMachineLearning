#!/bin/bash

# HAM
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/ham/beck-s.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/ham/farmer-d.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/ham/kaminski-v.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/ham/kitchen-l.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/ham/lokay-m.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/ham/williams-w3.tar.gz

# SPAM
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/spam/BG.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/spam/GP.tar.gz
wget http://www.aueb.gr/users/ion/data/enron-spam/raw/spam/SH.tar.gz

# Extract all
for a in `ls -1 *.tar.gz`; do tar -zxf $a; done

# Delete all tars
rm *.tar.gz

