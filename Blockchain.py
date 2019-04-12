import hashlib, json
block_genesis = {
 'prev_hash': None,
 'transactions' :[1, 3, 4, 2]
}
block_2 = {
 'prev_hash': None,
 'transactions': [3, 3, 3, 8, 7, 12]
}
block_3 = {
 'prev_hash': None,
 'transactions': [3, 4, 4, 8, 34, 7]
}
def hash_blocks(blocks):
 prev_hash = None
 for block in blocks:
  block['prev_hash'] = prev_hash
  block_serialized = json.dumps(block, sort_keys=True).encode('utf-8')
  block_hash = hashlib.sha256(block_serialized).hexdigest()
  prev_hash = block_hash
 return prev_hash
print('Original hash')
print(hash_blocks([block_genesis, block_2, block_3]))
print('Tampering the data')
block_genesis['transactions'][0] = 3
print('After being tampered')
print(hash_blocks([block_genesis, block_2, block_3]))