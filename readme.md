
# NeruialRG again

reimplementation solves all problems!(almost)

python ./benchmark.py

python ./main.py -epochs 5000 -folder ./opt/16Ising -batch 512 -cuda 2 -nlayers 10 -nmlp 3 -nhidden 64 -L 16 -savePeriod 100

python ./plot.py -folder ./opt/16Ising

python ./correlation.py -folder ./opt/16Ising -all

python ./paperPlot/lossplot.py -folder ./opt/16Ising -show -Lexact -592.875483222 -Hexact 0.544445