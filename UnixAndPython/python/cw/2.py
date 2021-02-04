


word_dict = {}
not_words = [",", ".", "!", "?"]

num_of_sen = 0
with open("text.txt", "r") as fin:
    for line in fin:
        sens = line.split(".")
        num_of_sen += len(sens) - 1  # count only left-side sentence
        line = line.replace(".", "").replace(",", "").replace(":", "").replace("!", "").replace("?", "")
        for word in line.split():
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        


sorted_dict = {k: word_dict[k] for k in sorted(word_dict, key=word_dict.get, reverse=True)}
iterator = iter(sorted_dict.items())
for i in range(10):
    print(f"{i+1}:", next(iterator))
    
print("Number of sentences:", num_of_sen)


def find_word(sword, text_path):
    word_dict = {}
    ans = []
    with open(text_path, "r") as fin:
        for line in fin:
            line = line.replace(".", "").replace(",", "").replace(":", "").replace("!", "").replace("?", "")
            splited = line.split(sword)
            if len(splited) > 1:
                for i in range(1, len(splited)):
                    next_word = splited[i].split()[0]
                    if next_word in word_dict:
                        word_dict[next_word] += 1
                    else:
                        word_dict[next_word] = 1
    
    sorted_dict = {k: word_dict[k] for k in sorted(word_dict, key=word_dict.get, reverse=True)}
    iterator = iter(sorted_dict.items())
    for _ in range(3):
        ans.append(next(iterator))
    return ans
    
# Без вероятности ибо время
            
print(find_word("nostrud", "text.txt"))