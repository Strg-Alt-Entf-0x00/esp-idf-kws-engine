#ifndef KWS_CONSOLE_DEV_CONFIG_H
#define KWS_CONSOLE_DEV_CONFIG_H

#include "esp_kws.h"
#include "esp_pulsevad.h"

#define KWS_SAMPLE_RATE_HZ             16000
#define KWS_BITS_PER_SAMPLE            16
#define KWS_NUM_CHANNELS               1
#define KWS_BYTES_PER_SAMPLE           2

#define FS_MOUNT_POINT_SD              "/sd"
#define MODEL_FILE_EXTENSION           ".espm"
#define WAV_FILE_EXTENSION             ".wav"

#define MAX_FILE_PATH                  256
#define MAX_FILENAME                   64

#define CONSOLE_MAX_COMMAND_LINE       256
#define CONSOLE_PROMPT                 "kws_dev> "
#define CONSOLE_HISTORY_SIZE           30

#endif
