import sys, math, threading

input = sys.stdin.readline
print2 = lambda x: sys.stdout.write(str(x) + "\n")
# sys.setrecursionlimit(1 << 30)
# threading.stack_size(1 << 27)
# main_threading = threading.Thread(target=main),
# main_threading.start()
# main_threading.join()


class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.word = None


class Trie:
    def __init__(self, m):
        self.root = self.getNode()
        self.found = ["No"] * m

    def getNode(self):
        return TrieNode()

    def get_char(self, ch):
        return ord(ch) - ord("a")

    def insert(self, key, idx):
        pCrawl = self.root
        for c in key:
            index = self.get_char(c)
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]
        pCrawl.word = idx


    def parse_text(self, text):
        for i in range(len(text)):
            v = self.root
            for j in range(i, len(text)):
                v = v.children[self.get_char(text[j])]
                if v:
                    if v.word:
                        self.found[v.word] = "Yes"
                else:
                    break


def main():
    text = input().strip()
    m = int(input().strip())
    trie = Trie(m)
    words = [None] * m

    for i in range(m):
        words[i] = input().strip()
        trie.insert(words[i], i)
        
            
    trie.parse_text(text)
    print(*trie.found, sep="\n")
        
    
    
    
    
    
    
    


main()
