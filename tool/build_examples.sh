#!/bin/bash
rm -f *.sw *.c
# ./bin/stenCC ../examples/$@/stencil_$@.dsl --enable-vector=4 --emit=sw > stencil_$@.sw 2>&1
./bin/stenCC ../examples/$@/stencil_$@.dsl --emit=sw > stencil_$@.sw 2>&1
python3 translate.py stencil_$@.sw
