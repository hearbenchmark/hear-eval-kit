Docker files for downstream evaluation, CUDA 11.2..
GCP CUDA is 11.0, but we use CUDA 11.2 to get latest TF

This includes torch 1.9 and tf 2.6.0.  We use 1.19.2 numpy for tf 2.6.0
We use a simple hack to get tf 2.4.2 to play nice with CUDA 11.2:
(https://medium.com/mlearning-ai/tensorflow-2-4-with-cuda-11-2-gpu-training-fix-87f205215419)
   ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11 \
       /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.10
   LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.2/targets/x86_64-linux/lib"

If you want to rebuild these dockers from the repository root:

./docker/build.sh to create and push.
