def mcp_neuron(inputs, weights, threshold):
    summation = sum(i * w for i, w in zip(inputs, weights))
    return 1 if summation >= threshold else 0

def AND(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], 2)

def OR(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], 1)

def NOT(x1):
    return mcp_neuron([x1], [-1], 0)

def NOR(x1, x2):
    return mcp_neuron([x1, x2], [-1, -1], 0)

def XOR(x1, x2):
    return x1 ^ x2

print("AND Gate:")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {AND(*x)}")

print("\nOR Gate:")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {OR(*x)}")

print("\nNOT Gate:")
for x in [0, 1]:
    print(f"{x} -> {NOT(x)}")

print("\nNOR Gate:")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {NOR(*x)}")

print("\nXOR Gate:")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {XOR(*x)}")
