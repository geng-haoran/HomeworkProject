#include<stdio.h>
# define NUM_SIZE 3
int main(int args, char *argv[])
{
    printf("cpp version\n");
    // static 
    int nums[NUM_SIZE]= {1,2,6};
    // nums = *;
    nums[0] = 0;
    nums[2] = 9;
    for(int i=0; i<3; i++)
    {
        printf("Num at index %d is %d\n", i, nums[i]);
    }
    char *str = "abcd";
    printf("My string is %s\n", str);
    return 0;
}