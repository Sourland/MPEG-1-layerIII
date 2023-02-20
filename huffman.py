import heapq
from collections import defaultdict
import numpy as np

codes = {}

class Node:
    def __init__(self, probability, symbol=None, left=None, right=None):
        self.probability = probability
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

    def __lt__(self, other):
        return self.probability < other.probability


def get_symbol_probabilities(symbols: np.ndarray) -> np.ndarray:
    probabilities = dict()
    for symbol in symbols:
        if probabilities.get(symbol) is None:
            probabilities[symbol] = 1 / len(symbols)
        else:
            probabilities[symbol] += 1 / len(symbols)
    return probabilities


def print_codes(root, sequence):
    if root is None:
        return
    if root.data != '$':
        print(root.data, ":", sequence)
    printCodes(root.left, sequence + "0")
    printCodes(root.right, sequence + "1")


def store_codes(root, sequence):
    if root is None:
        return
    if root.symbol != '$':
        codes[root.symbol] = sequence
    store_codes(root.left, sequence + "0")
    store_codes(root.right, sequence + "1")


def huffman_encode(data):
    global minHeap
    probabilities = get_symbol_probabilities(data)
    for key in probabilities:
        minHeap.append(Node(probabilities[key], key))

    heapq.heapify(minHeap)
    while len(minHeap) != 1:
        left = heapq.heappop(minHeap)
        right = heapq.heappop(minHeap)
        top = Node(probability=left.probability + right.probability, symbol='$')
        top.left = left
        top.right = right
        heapq.heappush(minHeap, top)
    return store_codes(minHeap[0], "")


def decode_file(root, s):
    curr = root
    ans = []
    n = len(s)
    for i in range(n):
        if s[i] == '0':
            curr = curr.left
        else:
            curr = curr.right

        # reached leaf node
        if curr.left is None and curr.right is None:
            ans.append(curr.symbol)
            curr = root
    return ans


minHeap = []
arr = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
huffman_encode(arr)
print(codes)
encoded = ""
for symbol in arr:
    encoded += codes[symbol]

print(encoded)
print(decode_file(minHeap[0], encoded))