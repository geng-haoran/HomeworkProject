#include<stdio.h>
#include<stdlib.h>
#include<string.h>
struct node
{ 
    int x; int y;
};


int main(int argc, char **argv)
{
    printf("%s %d\n", argv[0], argc);
    struct node n1;
    struct node *n2 = (struct node*)malloc(sizeof(struct node));

    printf("%d, %d\n", (*n2).x, n2->y);
    n2 -> x = 1;
    n2 -> y = 4;
    printf("%d, %d\n", (n2)->x, n2->y);

}