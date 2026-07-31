@echo off
title Analyseur BO - NLP Juridique Marocain
cd /d "%~dp0"
echo ============================================
echo   Analyseur de Bulletins Officiels
echo   NLP Juridique Marocain
echo ============================================
echo.
python lanceur_web.py
echo.
echo Appuyez sur une touche pour fermer...
pause
