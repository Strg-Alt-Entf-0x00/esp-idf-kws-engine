idf_component_register(
    SRCS
        "src/esp_kws.c"
        "src/kws_frontend.c"
        "src/kws_nn.c"
        "src/loaders/kws_espm_loader_core.c"
        "src/loaders/kws_espm_loader_vfs.c"
    INCLUDE_DIRS
        "include"
    PRIV_INCLUDE_DIRS
        "src"
    REQUIRES
        esp-dsp
        esp_timer
        esp-idf-model-manager
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Ofast
    -funroll-loops
    -fno-strict-aliasing
    -falign-functions=32
    -falign-loops=16
)

set_source_files_properties(src/loaders/kws_espm_loader_core.c PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math"
)
