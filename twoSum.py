class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        p = nums[:]
        p.sort()
        r = len(nums)-1
        l =0 
        
        for i in range(len(nums)):
          if(p[l] + p[r]<target): 
            l += 1
          elif (p[l] + p[r]>target): 
            r -=1
          else :
            first_num = p[l]
            second_num = p[r]
        
        first = True
        for i in range(0,len(nums)):
          if(nums[i]==first_num and first==True):
            first_index = i
            first = False
          elif (nums[i]==second_num):
            second_index = i
        

        return [first_index,second_index]

s = Solution()        
print(s.twoSum([3,3],6))        