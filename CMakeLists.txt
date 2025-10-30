cmake_minimum_required(VERSION 3.16)
project(ml_gestures_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- OpenCV (Homebrew path) ----
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ---- TensorFlow C (Homebrew: /opt/homebrew/lib & include) ----
find_library(TF_LIBRARY tensorflow REQUIRED PATHS /opt/homebrew/lib)

add_executable(ml_gestures src/main.cpp)
target_include_directories(ml_gestures PRIVATE ${OpenCV_INCLUDE_DIRS} /opt/homebrew/include)
target_link_libraries(ml_gestures
  ${OpenCV_LIBS}
  ${TF_LIBRARY}
)

# Ensure the app finds dylibs at runtime on macOS
if(APPLE)
  set_target_properties(ml_gestures PROPERTIES
    BUILD_RPATH "/opt/homebrew/lib"
    INSTALL_RPATH "/opt/homebrew/lib")
endif()