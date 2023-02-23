#include <stdio.h>
#include <string.h>
#include<stdlib.h>

// Returns a malloced copy of the string
char *str_copier(char *str) {
    int length = strlen(str);

    char *copied = (char*)malloc(sizeof(char) * (length + 1));
    // Consider the following commented out code
    // char copied[length + 1];

    strcpy(copied, str);
    return copied;
}


int main(int argc, char *argv[]) {
    // Declare a string
    char *str = "Help";
    char *new_str = str_copier(str);
    printf("Copied str: %s\n", new_str);
    new_str = "123";
    // What's missing here? free(str);
    printf("New: %s\n", new_str);
    printf("New: %s\n", str);

    return 0;
}
