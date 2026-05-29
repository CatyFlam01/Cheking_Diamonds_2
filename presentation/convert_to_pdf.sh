#!/bin/bash
# Конвертация Markdown презентации в PDF

pandoc presentation/presentation.md \
  -o presentation/Diamonds_Project_Presentation.pdf \
  --from markdown \
  --to pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=12pt

echo "Презентация сохранена: presentation/Diamonds_Project_Presentation.pdf"
