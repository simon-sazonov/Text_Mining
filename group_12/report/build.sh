#!/bin/bash
# Build report_12.pdf from report_12.md (pandoc -> typst). Run from group_12/report/.
set -e
~/.local/bin/pandoc report_12.md -t typst -o .body.typ
cat preamble.typ .body.typ > .full.typ
~/.local/bin/typst compile .full.typ report_12.pdf --root .
rm .body.typ .full.typ
echo "report_12.pdf built"
