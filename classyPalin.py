import numpy 
class Solution:
    def __init__(self, string):
      self.x = string
      self.x = 's' + self.x
      self.m = numpy.zeros(shape=(len(self.x),len(self.x)))
    def longestPalin(self):
      
        
        
        for i in range(1,len(self.x)):
            self.m[i][i] = 1
        
        for substring_length in range(1,len(self.x)-1):
            for i in range(1,len(self.x)):
                if i + substring_length > len(self.x)-1 :
                    break;
                row = i
                column = i + substring_length
                if self.x[row]==self.x[column]:
                    self.m[row][column] = self.m[row+1][column-1] + 2
                else : 
                    self.m[row][column] = max(self.m[row][column-1],self.m[row+1][column]) 

    def make(self, i,j):
        if(i==j):
            print(self.x[i]) 
        elif i<j:
            if self.x[i]==self.x[j]:
                print(self.x[i])
                self.make(i+1,j-1)
                print(self.x[i])
            elif self.m[i][j-1]>self.m[i+1][j]:
                self.make(i,j-1)
            else:
                self.make(i+1,j)
        else:
            return ''

c = Solution("aibohphobia")
c.longestPalin()

print(c.make(1,len("aibohphobia")))