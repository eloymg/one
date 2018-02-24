from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto.Util import number

key = b'Sixteen byte key'
iv = number.getRandomInteger(128)
ctr = Counter.new(128,initial_value=iv)
print(str(iv))
cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
msg = cipher.encrypt(b'a')

ctr = Counter.new(128,initial_value=iv)
cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
