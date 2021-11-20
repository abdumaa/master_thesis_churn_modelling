#!/bin/bash
cd /Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling/
# 1. Copy thesis.tex in new folder "tex/"
mkdir tex/
cp thesis.tex tex/thesis.tex
# 2. Create pdf
cd tex/
pdflatex thesis.tex
# 3. Copy pdf in thesis directory
yes | cp -rf thesis.pdf ../thesis.pdf
# 4. Clean up
cd ..
rm -r tex/
cd /Users/abdumaa/Desktop/Uni_Abdu/Master/Masterarbeit/master_thesis_churn_modelling