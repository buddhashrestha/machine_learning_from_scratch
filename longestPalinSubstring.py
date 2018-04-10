import numpy 
class Solution:
    def longestPalindrome(self,x):
        m = numpy.zeros(shape=(len(x),len(x)))
        for i in range(0,len(x)):
            m[i][i] = 1
        start = 0
        end = 0
        for substring_length in range(1,len(x)):
            for i in range(0,len(x)):
                if i + substring_length >= len(x):
                    break;
                row = i
                column = i + substring_length
                if x[row]==x[column]:
                    if m[row+1][column-1] == 1 or (row+1>column-1):
                        m[row][column] = 1
                        start = row
                        end = column
                else : 
                    m[row][column] = 0
        return  x[start:end+1]


print(Solution().longestPalindrome("abababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababa"))

#aibohphobia