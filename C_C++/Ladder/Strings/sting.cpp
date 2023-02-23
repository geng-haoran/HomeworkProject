#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(int argc, char **argv)
{
    char *str = "hello world";
    printf("%c\n", str[0]);
    printf("%s\n", str);
    // print
    printf("%d\n", strlen(str));
    char* copied_str = (char *)malloc(sizeof(char*)*12);
    strcpy(copied_str, str);
    printf("%s\n", copied_str);
    printf("%d\n", strlen(copied_str));
    char *n = copied_str;
    
    printf("%s\n", copied_str);
    printf("%s\n", str);

    return 0;
}