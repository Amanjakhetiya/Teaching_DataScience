@echo off
for /r %%i in (Main_Workshop_AWSMath*.tex) do texify -cp %%i
