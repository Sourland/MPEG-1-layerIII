import heapq
from collections import defaultdict
import numpy as np


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


def store_codes(root, sequence, frame_codes):
    if root is None:
        return

    if root.left is None and root.right is None and root.symbol == '$':
        frame_codes[root.symbol] = "0"
        return

    if root.symbol != '$':
        frame_codes[root.symbol] = sequence
    store_codes(root.left, sequence + "0", frame_codes)
    store_codes(root.right, sequence + "1", frame_codes)


def build_heap(heap, probabilities):
    for key in probabilities:
        heap.append(Node(probabilities[key], key))

    heapq.heapify(heap)
    while len(heap) != 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        top = Node(probability=left.probability + right.probability, symbol='$')
        top.left = left
        top.right = right
        heapq.heappush(heap, top)


def huffman_build_codes(heap, probabilities):
    frame_codes = {}
    build_heap(heap, probabilities)
    store_codes(heap[0], "", frame_codes)
    return frame_codes


def huffman_encode(data):
    heap = []
    probabilities = get_symbol_probabilities(data)
    codes = huffman_build_codes(heap, probabilities)
    encoded = ""
    for symbol in data:
        encoded += codes[symbol]
    return probabilities, encoded


def huffman_decode(encoded, probabilities):
    heap = []
    build_heap(heap, probabilities)
    return decode_sequence(heap[0], encoded)


def decode_sequence(root, s):
    if root.left is None and root.right is None:
        return root.symbol
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
    return np.array(ans)


# heap = []
# arr = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
# arr = np.array([1151, 3, 3, 2, 2, 2])
# probabilites, encoded = huffman_encode(arr)
#
# print(encoded)
# print(probabilites)
#
# decoded = huffman_decode(encoded, probabilites)
# print(decoded)
