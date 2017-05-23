#bin/bash!
for N in {1..8}; do ../../util/align-dlib.py ./faces/originalData/ align outerEyesAndNose ./faces/alignedData --size 96 & done
../../batch-represent/main.lua -outDir ./faces/trainingData -data ./faces/alignedData
