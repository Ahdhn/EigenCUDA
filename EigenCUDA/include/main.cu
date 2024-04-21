
#include "gtest/gtest.h"

#include "test_eigen_decomp.h"
#include "test_svd.h"


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
