# Hebbian training
python ./examples/hebbian.py --json params/mnist.json --device cpu

# Supervised training
python ./examples/supervised.py

# Free Energy training
python ./examples/free_energy.py --json params/mnist_FEP.json --device cpu

# Test
python ./examples/test.py