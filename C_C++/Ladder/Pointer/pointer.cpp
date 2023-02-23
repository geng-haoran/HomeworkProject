#include<stdio.h>

int modify(int z)
{
    z = 5;
    return 0;
}

int modify_(int *z)
{
    *z = 5;
    return 0;
}

int main(int argc, char* argv[])
{
    int x = 4;
    int *y = &x;
    printf("pointer %ld\n", y);
    printf("pointer's value %d\n", *y);
    printf("pointer's pointer %d\n", &y);
    *y = 6;
    printf("pointer %ld\n", y);
    printf("pointer's value %d\n", *y);
    printf("pointer's pointer %d\n", &y);
    modify(x);
    printf("pointer %ld\n", y);
    printf("pointer's value %d\n", *y);
    printf("pointer's pointer %d\n", &y);
    modify_(y);
    printf("pointer %ld\n", y);
    printf("pointer's value %d\n", *y);
    printf("pointer's pointer %d\n", &y);


}