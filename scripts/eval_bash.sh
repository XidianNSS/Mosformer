export OMP_NUM_THREADS=1

cd ..

if [ ! -d "build" ]; then
    mkdir build
fi

cd ./build || { echo "Failed to enter build directory"; exit 1; }

# 运行 cmake 和 make
cmake .. || { echo "cmake failed"; exit 1; }

make -j || { echo "make failed"; exit 1; }

if [ $# -eq 1 ]; then
  ./mosformer $1 0 &

  ./mosformer $1 1 &

  ./mosformer $1 2 &

elif [ $# -gt 1 ]; then
  ./mosformer $1 $2
fi

wait

echo "Completed."