class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        zipped = zip(*strs)
        pref_string = ''
        for zipp in zipped:
            if not len(set(zipp)) == 1:
                break
            pref_string = pref_string + zipp[0]
        return pref_string


print(Solution().longestCommonPrefix(["a","b","c"]))


### IMPROVEMENT:


        

# class Solution(object):
#     def longestCommonPrefix(self, strs):
#         """
#         :type strs: List[str]
#         :rtype: str
#         """
#         if len(strs)==0:
#             return ''
#         if len(strs)==1:
#             return strs[0]
#         strs.sort()
#         pre=strs[0]
#         for i in list(range(len(strs))):
#             if pre=='':
#                 break
#             while pre!=strs[i][:len(pre)]:
#                 pre=pre[:-1]
#                 if pre=='':
#                     break
#         return pre