def quicksort(a_list):
    """Hoare partition scheme, see https://en.wikipedia.org/wiki/Quicksort"""
    def _quicksort(a_list, low, high):
        # must run partition on sections with 2 elements or more
        if low < high: 
            p = partition(a_list, low, high)
            _quicksort(a_list, low, p)
            _quicksort(a_list, p+1, high)
    def partition(a_list, low, high):
        pivot = a_list[low]
        while True:
            while a_list[low] < pivot:
                low += 1
            while a_list[high] > pivot:
                high -= 1
            if low >= high:
                return high
            a_list[low], a_list[high] = a_list[high], a_list[low]
            low += 1
            high -= 1
    _quicksort(a_list, 0, len(a_list)-1)
    return a_list

print(quicksort([2,5,3,0,3,2,3,0,3]))