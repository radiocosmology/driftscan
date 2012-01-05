#!/bin/bash

datadir=$(dirname $0)

data_url="http://www.cita.utoronto.ca/~jrs65/cyl_skydata.tar.gz"

echo "Downloading data file....."

curl $data_url | tar xzv

RETVAL=$?
[ $RETVAL -eq 0 ] && echo "Done."
[ $RETVAL -ne 0 ] && echo "Download failed, you're on your own.\nTry fetching $data_url manually."



