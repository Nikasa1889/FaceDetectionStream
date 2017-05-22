#bin/bash!
for N in {1..8}; do ../../util/align-dlib.py ./data/ align outerEyesAndNose ./alignedData --size 96 & done
