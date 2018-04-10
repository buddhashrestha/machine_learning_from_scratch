import numpy 
class Solution:
    def longestPalin(self,x):
        x = 's' + x
        m = numpy.zeros(shape=(len(x),len(x)))
        for i in range(1,len(x)):
            m[i][i] = 1
        
        for substring_length in range(1,len(x)-1):
            for i in range(1,len(x)):
                if i + substring_length > len(x)-1 :
                    break;
                row = i
                column = i + substring_length
                if x[row]==x[column]:
                    m[row][column] = m[row+1][column-1] + 2
                else : 
                    m[row][column] = max(m[row][column-1],m[row+1][column]) 
        

        ##backtracking
        i = 1
        j = len(x) - 1
        print("Printing yo!")
        palinWord = ''
        while(True):
            if len(palinWord) == m[1][len(x)-1]:
                break
            if i == j:
                palinWord =palinWord[:len(palinWord)/2] + x[i] + palinWord[len(palinWord)/2:]
                break
            if (m[i+1][j]==m[i][j-1] and m[i][j]==m[i+1][j-1] + 2):
                palinWord = palinWord[:len(palinWord)/2] + x[i] + x[j] + palinWord[len(palinWord)/2:]
                i = i + 1
                j = j - 1
            elif (m[i][j-1]==m[i][j]):
                j = j - 1
            elif (m[i+1][j]==m[i][j]):
                i = i + 1
        return palinWord
print(Solution().longestPalin("aibohphobia"))