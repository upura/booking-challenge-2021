#!/bin/bash

for i in `seq 0 24`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/1iewPD9nKSa1bl-fywGfIBGlJQQalpJsj?authuser=1
  sleep 3600
done
