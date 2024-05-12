window_size = 11
check = [i for i in range(0,22)]
guess = []
for i in range(0,len(check)-window_size+1):
    guess.append(f"{check[i]}-{check[i+window_size-1]}")
print(guess)
print(len(guess))
print(check[window_size-1:])
