set(EMBED_FILES
    "${CMAKE_CURRENT_LIST_DIR}/../../../models/espm/wake_ghost.espm"
    "${CMAKE_CURRENT_LIST_DIR}/../../../models/espm/cmd_smarthome.espm"
)

set(TEST_AUDIO_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../.data/test_audio")
set(HAVE_TEST_AUDIO FALSE)
if(EXISTS "${TEST_AUDIO_DIR}/test_wake_espira.wav" AND
   EXISTS "${TEST_AUDIO_DIR}/test_cmd_licht_an.wav" AND
   EXISTS "${TEST_AUDIO_DIR}/test_background.wav")
    list(APPEND EMBED_FILES
        "${TEST_AUDIO_DIR}/test_wake_espira.wav"
        "${TEST_AUDIO_DIR}/test_cmd_licht_an.wav"
        "${TEST_AUDIO_DIR}/test_background.wav"
    )
    set(HAVE_TEST_AUDIO TRUE)
endif()

idf_component_register(
    SRCS "main.cpp" "cmd_kws_cli.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES ${EMBED_FILES}
    REQUIRES
        esp-idf-kws-engine
        esp-idf-pulsevad
        esp-idf-model-manager
        board_audio
        console
        nvs_flash
        fatfs
        sdmmc
        esp_driver_sdmmc
        esp_driver_uart
    PRIV_REQUIRES
        driver
        esp_timer
        esp_pm
)

if(HAVE_TEST_AUDIO)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE CONFIG_KWS_EMBED_TEST_AUDIO=1)
endif()
