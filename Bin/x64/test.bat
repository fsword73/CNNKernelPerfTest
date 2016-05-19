REM   only 64 threads

set TEST=CNNBench.exe  
for %%i in (3 5 7 9 11 13 15 17 19 21 ) do (
   %TEST% -dim 1 -gx 64 -gy 1 -lx 64 -ly 1 -f %%i -c1 2048 -c2 %%i
)


REM   only 2048x2048 threads
FOR %%i IN (3 5 7 9 11 13 15 17 19 21 ) DO (
   %TEST% -dim 1 -gx 4194304 -gy 1 -lx 64 -ly 1 -f %%i -c1 2048 -c2 %%i
)


REM   only 4096*4096 threads
FOR %%i IN (3 5 7 9 11 13 15 17 19 21 ) DO (
   %TEST% -x 4096 -y 4096 -dim 1 -gx 16777216 -gy 1 -lx 64 -ly 1 -f %%i -c1 4096 -c2 %%i
)
