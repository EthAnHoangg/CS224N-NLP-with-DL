import numpy as np

a = np.array([1,2,4,8,20])
# sum(a) = 35

m = 42
w = 37

#--- calculating the values of public key using m and w------
pk = np.mod((w * a), m)
print("Public key:", pk)
# array([37, 32, 22,  2, 26])
#============== Encryption ================
plain_text = np.array([[1,0,0,1,0],
                        [1,1,0,1,1],
                        [0,1,0,1,0]])

cipher_text = np.dot(plain_text, pk)
print("Cipher_text:", cipher_text)
# [ 39 97 34]

#============== Decryption ================
# receiver received cipher_text = [8 44 13]
# also knew the value of m = 42, w = 37
# first we need to find the value of w^-1
# w * (w^-1) mod (m) = 1, which means that (w^-1 * 37 mod 42 = 1)

def knapsack_solver(word, a):
    res = [0] * len(a)
    n = len(a)
    for i in range (n-1, -1, -1):
        if word - a[i] >= 0:
            res[i] = 1
            word -= a[i]
        else:
            res[i] = 0

    return res

def decryption(cipher_text, m, w, a):
    for i in range (1000):
        if np.mod(i * m + 1, w) == 0:
            w_prime = (i * m + 1)/w
            break
    cipher_text = np.mod(cipher_text * w_prime, m)
    print(cipher_text)
    plain_text = []
    for word in cipher_text:
        plain_text.append((knapsack_solver(word, a)))
    return plain_text


print(decryption(cipher_text, m, w, a))