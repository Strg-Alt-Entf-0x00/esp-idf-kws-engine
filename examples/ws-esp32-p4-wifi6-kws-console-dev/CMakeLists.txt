cmake_minimum_required(VERSION 3.16)

# Register parent esp-idf-kws-engine directory
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../.."
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ws-esp32-p4-wifi6-kws-console-dev)
