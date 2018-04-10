from collections import deque
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        d = {
            '(':")",
            '[':"]",
            '{':"}"
        }
        e = deque()
        for ch in s:
            if ch in d:       #key exists
                e.append(ch)
            else: 

                if e and ch == d[e[-1]]:
                    e.pop()
                else:
                    return False

        return len(e) == 0
                
print(Solution().isValid('(])'))