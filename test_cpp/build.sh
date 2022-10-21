#!/usr/bin/env bash
if [ ! -d "build/" ];then
  mkdir "build"
else
  echo "exist build"
fi
cd build
cmake ..
make -j4
sleep 1
#./main
#./kalman_test
./openmp_test

