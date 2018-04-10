class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        dict = {'I':1,
                'V':5,
                'X':10,
                'L':50,
                'C':100,
                'D':500,
                'M':1000,
                'F':0
               }
        
        s = s + 'F'
        sum = 0
        prev = 'F'
        for ch in s:
            num = dict[ch]
            if num > dict[prev] and dict[prev]!=0:
                num = num - dict[prev]
                prev = 'F'
            else:
                num = dict[prev]
                prev = ch
            sum = sum + num
            
        return sum
    

print(Solution().romanToInt("MCMXCVI"))