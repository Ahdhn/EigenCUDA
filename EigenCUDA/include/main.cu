#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"



__global__ void exec_kernel()
{

}

TEST(Test, exe)
{
    exec_kernel<<<1, 1>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);    
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
