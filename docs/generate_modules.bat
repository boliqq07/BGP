
@echo on
set path=D:\Anaconda3;D:\Anaconda3\Library\bin;D:\Anaconda3\Scripts;D:\Anaconda3\condabin;%path%
path

sphinx-apidoc -f -o ./source/modules ../bgp

make clean

pause

exit