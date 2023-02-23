#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[])
{
    int *arr2 = (int *) malloc(sizeof(int) * 3);
    printf("arr2: %x\n", arr2);
    for (int i=0; i<3; i++)
    {
        *(arr2+i) = i;
        printf("%d %x\n", i ,arr2+i);
    }

    return 0;
}