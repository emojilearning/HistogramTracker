cmake_minimum_required(VERSION 3.6)
project(HistogramTracker)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CmakeModules/)
add_compile_options(-O3)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)

file(GLOB_RECURSE SOURCE_FILES  ${PROJECT_SOURCE_DIR}/src/*.c* ${PROJECT_SOURCE_DIR}/ThirdParty/*.c*)
file(GLOB_RECURSE HEADERS  ${PROJECT_SOURCE_DIR}/include/*.h* ${PROJECT_SOURCE_DIR}/ThirdParty/*.h*)

#set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/include/*.h)

# add_executable(HistogramTracker ${SOURCE_FILES} ${HEADERS})

INCLUDE_DIRECTORIES(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/ThirdParty
        ${EIGEN3_INCLUDE_DIR}
        ${GLOG_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
)
function(assign_source_group)
foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE "${_source}")
        file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
    else()
        set(_source_rel "${_source}")
    endif()
    get_filename_component(_source_path "${_source_rel}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${_source_path_msvc}" FILES "${_source}")
endforeach()
endfunction(assign_source_group)

function(my_add_executable)
foreach(_source IN ITEMS ${ARGN})
    assign_source_group(${_source})
endforeach()
add_executable(${ARGV})
endfunction(my_add_executable)
my_add_executable(HistogramTracker ${SOURCE_FILES} ${HEADERS} )

SET_TARGET_PROPERTIES(HistogramTracker PROPERTIES    RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}/bin")

ADD_DEFINITIONS(/D_USE_MATH_DEFINES )

LIST(APPEND LINK_LIBS ${OpenCV_LIBS} ${GLOG_LIBRARIES} ceres)
TARGET_LINK_LIBRARIES(HistogramTracker ${LINK_LIBS})
