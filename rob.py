import operator

class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s = [0] * len(nums)
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 0:
            return 0
        s[0] = nums[0]
        s[1] = nums[1]
        for i in range(2,len(nums)):
            if i-3<0:
                prev_sum = s[i-2]
            else:
                prev_sum = max(s[i-2],s[i-3])
            s[i] = nums[i] + prev_sum 

        index, value = max(enumerate(s), key=operator.itemgetter(1))
        
        return value


print(Solution().rob([]))