class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        num = 0
        c = x
        neg = False
        if x < 0:
            return False
        else :
            x = int (x)
        rem = x % 10
        while(x):
            num = num * 10 + rem
            x = x / 10
            rem = x % 10
        if(abs(num) > (2 ** 31 - 1)):
            return False
        if (num==c) : return True
        else : return False
            
print(Solution().isPalindrome(1))