#pragma once
#include <Eigen/Dense>

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

__device__ __host__ void test_svd()
{
    Eigen::Matrix3d matrix;
    matrix << 1, 2, 3,  //
        4, 5, 6,        //
        7, 8, 9;

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


    Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV>
        svd(matrix);


    Eigen::Vector3d singular_values        = svd.singularValues();
    Eigen::Matrix3d left_singular_vectors  = svd.matrixU();
    Eigen::Matrix3d right_singular_vectors = svd.matrixV();


    printf("\n singular_values = {%f, %f, %f}",
           singular_values(0),
           singular_values(1),
           singular_values(2));

    // verify
    Eigen::Matrix3d reconstructed_matrix = left_singular_vectors *
                                           singular_values.asDiagonal() *
                                           right_singular_vectors.transpose();

    printf("\n Reconstr = {%f, %f, %f || %f, %f, %f || %f, %f, %f}\n",
           reconstructed_matrix(0, 0),
           reconstructed_matrix(0, 1),
           reconstructed_matrix(0, 2),
           reconstructed_matrix(1, 0),
           reconstructed_matrix(1, 1),
           reconstructed_matrix(1, 2),
           reconstructed_matrix(2, 0),
           reconstructed_matrix(2, 1),
           reconstructed_matrix(2, 2));
}


__global__ void device_svd()
{
    printf("\n From the device");
    test_svd();
}

TEST(Test, SVD)
{
    device_svd<<<1, 1>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    printf("\n From the host");
    test_svd();
}