conda create --name=python3 python=3.8.16
# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

>>> import cv2 Traceback (most recent call last):   File "<stdin>", line 1, in <module>   File "/home/stw_22210240266/.conda/envs/myPytorch2/lib/python3.8/site-packages/cv2/__init__.py", line 181, in <module>     bootstrap()   File "/home/stw_22210240266/.conda/envs/myPytorch2/lib/python3.8/site-packages/cv2/__init__.py", line 153, in bootstrap     native_module = importlib.import_module("cv2")   File "/home/stw_22210240266/.conda/envs/myPytorch2/lib/python3.8/importlib/__init__.py", line 127, in import_module     return _bootstrap._gcd_import(name[level:], package, level) ImportError: libGL.so.1: cannot open shared object file: No such file or directory

Solution : https://blog.csdn.net/Sinlair/article/details/125869756

pip install kornia

安装nvitop，没网，源码安装 -> 报错，没有cmake，源码安装 -> 依然报错 

cmake安装： https://blog.csdn.net/Man_1man/article/details/126467371

CMake Error at Utilities/cmcurl/CMakeLists.txt:645 (message):  Could not find OpenSSL.  Install an OpenSSL development package or  configure CMake with -DCMAKE_USE_OPENSSL=OFF to build without OpenSSL.   

依然报错：https://blog.csdn.net/xiachong27/article/details/115308416