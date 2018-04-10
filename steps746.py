import numpy
class Solution:
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        n = len(cost) 
        s = [0] * n
        if n == 0 or n == 1:
            return 0
        min1 = cost[0]
        min2 = cost[1]
        for i in range(2,len(cost)):
            min1,min2 = min2,min(min1,min2) + cost[i]
        return min(min1,min2)

print(Solution().minCostClimbingStairs( [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))