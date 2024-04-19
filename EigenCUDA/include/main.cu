#include <Eigen/Dense>

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"



__device__ __host__ void test_func()
{
    Eigen::Matrix3d matrix;
    matrix << 1, 2, 3,  //
        4, 5, 6,        //
        7, 8, 9;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix);

    Eigen::Vector3d eigenvalues  = eigen_solver.eigenvalues();
    Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();


    printf("\n Original = {%f, %f, %f || %f, %f, %f || %f, %f, %f}",
           matrix(0, 0),
           matrix(0, 1),
           matrix(0, 2),
           matrix(1, 0),
           matrix(1, 1),
           matrix(1, 2),
           matrix(2, 0),
           matrix(2, 1),
           matrix(2, 2));


    printf("\n eigenvalues = {%f, %f, %f}",
           eigenvalues(0),
           eigenvalues(1),
           eigenvalues(2));

    // verify
    Eigen::Matrix3d reconstructed_matrix =
        eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

    printf("\n Reconstr = {%f, %f, %f || %f, %f, %f || %f, %f, %f}",
           reconstructed_matrix(0, 0),
           reconstructed_matrix(0, 1),
           reconstructed_matrix(0, 2),
           reconstructed_matrix(1, 0),
           reconstructed_matrix(1, 1),
           reconstructed_matrix(1, 2),
           reconstructed_matrix(2, 0),
           reconstructed_matrix(2, 1),
           reconstructed_matrix(2, 2));

    printf("\n");
}

__global__ void eigen_decomp()
{
    printf("\n From the device");
    test_func();
}

TEST(Test, exe)
{
    eigen_decomp<<<1, 1>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    printf("\n From the host");
    test_func();
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
