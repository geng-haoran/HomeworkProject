project(CUDA_MAT_MUL LANGUAGES CXX CUDA)

project(CUDA_MAT_MUL)
if(CUDA_ENABLE)
    enable_language(CUDA)
endif()

add_library(cudaMatMul cudaMatMul.cu cudaMatMul.h)
target_compile_features(cudaMatMul PUBLIC cxx_std_11)

add_executable(main main.cc)
target_link_libraries(main cudaMatMul)