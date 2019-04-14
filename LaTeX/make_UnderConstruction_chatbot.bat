@echo off
for /r %%i in (Main_Seminar_Chatbot*.tex) do texify -cp %%i
