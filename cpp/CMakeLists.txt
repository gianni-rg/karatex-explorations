cmake_minimum_required(VERSION 3.29.0)
project(karate_multiview_frames_exporter VERSION 0.1.0 LANGUAGES C CXX)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()


add_executable(karate_multiview_frames_exporter ${CMAKE_CURRENT_SOURCE_DIR}/src/karate_multiview_frames_exporter.cpp)
target_compile_features(karate_multiview_frames_exporter PRIVATE cxx_std_20)
target_include_directories(karate_multiview_frames_exporter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third-party/include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
