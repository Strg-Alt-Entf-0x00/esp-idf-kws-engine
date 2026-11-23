#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t register_kws_console_commands(void);
void play_boot_chime(void);

#ifdef __cplusplus
}
#endif
