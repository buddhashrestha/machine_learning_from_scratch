class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        
        num = 0
        neg = False
        if x < 0:
            neg = True
            x = int (x * (-1))
        else :
            x = int (x)
        rem = x % 10
        while(x):
            num = num * 10 + rem
            x = x / 10
            rem = x % 10
        if(abs(num) > (2 ** 31 - 1)):
            return 0
        if neg:
            num = num * (-1)
        return num

print(Solution().reverse(1534236469))