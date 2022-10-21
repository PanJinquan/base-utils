#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cl_info.h"
//#include "HelloWorld.h"
#include "Convolution.h"

int main() {
    //get_cl_info();
    //const char *cl_file = "../contrib/base_cl/kernel/HelloWorld_Kernel.cl";
    //const char *cl_file = "../contrib/base_cl/kernel/HelloWorld.cl";
    const char *cl_file = "../contrib/base_cl/kernel/Convolution.cl";
    test(cl_file);
    printf("finish\n");
    return 0;
}



