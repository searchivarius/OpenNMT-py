#!/usr/bin/env python
import sys

opt=sys.argv[1]
totDim=int(sys.argv[2])

qty=len(opt.split('-'))
print(totDim // qty)