#include "main.h"
#include "arm_math.h"
#include <math.h>
#include <string.h>
#include "audio_config.h"
#include "gfcc.h"
#include "spectrogram.h"
#include "nnmf.h"
#include "fvad.h"
#include "mfcc.h"

/* STM32 INIT */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_I2S2_Init(void);

/* HANDLES */
I2S_HandleTypeDef hi2s2;
UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_spi2_rx;

/* PARAMETERS */
#define SAMPLE_RATE        16000
#define CHUNK_SIZE         8000     // 0.5 sec
#define TOTAL_CHUNKS       6        // 3 sec total

#define FRAME_SIZE 1024
#define FRAME_HOP  512
#define FFT_SIZE   1024
#define MAX_BINS   (FFT_SIZE/2)    // 512

#define MAX_FRAMES ((CHUNK_SIZE - FRAME_SIZE) / FRAME_HOP + 1)   // = 14

/* ---------- VAD & SILENCE DETECTION ---------- */
#define NOISE_FLOOR_MULTIPLIER  1.2f    /* adjust if needed */
#define MIN_SPEECH_ENERGY       0.02f
#define VARIANCE_THRESHOLD      0.005f
#define VAD_FRACTION_REQUIRED   0.6f
#define CALIBRATION_CHUNKS      3

/* DEBUG options */
// #define DEBUG_VAD
// #define DEBUG_RAW   /* uncomment to see raw DMA values during calibration/recording */

/* LED pins – adjust to your board */
#define LED1_PIN   GPIO_PIN_1
#define LED1_PORT  GPIOA
#define LED2_PIN   GPIO_PIN_2
#define LED2_PORT  GPIOA

/* DMA buffer – now uint16_t because HAL uses 16-bit transfers even for 24-bit data */
#define DMA_BUF_SIZE        512     /* number of uint16_t entries */
static uint16_t dma_buffer[DMA_BUF_SIZE];

/* Ping‑pong audio buffers */
static float audio_ping[CHUNK_SIZE];
static float audio_pong[CHUNK_SIZE];
static volatile uint8_t active_buffer = 0;
static volatile uint32_t buffer_index = 0;
static volatile uint8_t chunk_ready = 0;

/* Spectrogram array */
static float spectrogram[MAX_FRAMES][MAX_BINS];
static uint32_t num_frames;

/* Processing buffers */
static float frame_buffer[FRAME_SIZE];
static float features[GFCC_NUM_COEFFS];
static float W[MAX_BINS];
static float H[MAX_FRAMES];
static float stats_w[8];
static float stats_h[8];

GFCC_Handle gfcc;

/* Control flags */
uint8_t recording = 0;
uint8_t chunk_count = 0;
uint8_t system_locked = 0;
static uint8_t calibrating = 0;

/* Button debouncing */
uint8_t prev_button = 0;
uint32_t last_press_time = 0;
uint8_t button_released = 0;

/* VAD handle */
static Fvad *vad_inst = NULL;

/* Noise floor calibration */
static float noise_floor = 0.0f;
static uint8_t noise_floor_calibrated = 0;

/* DC filter state */
static float x_prev = 0.0f;
static float y_prev = 0.0f;

/* Clipping detection */
static uint8_t clipping_reported = 0;

/* UART functions */
void uart_print(const char *s) {
    HAL_UART_Transmit(&huart2, (uint8_t*)s, strlen(s), HAL_MAX_DELAY);
}

void uart_print_int(int value) {
    char buf[16];
    int i = 0;
    if(value == 0) buf[i++] = '0';
    else {
        char rev[16];
        int j = 0, temp = value;
        if(temp < 0) { uart_print("-"); temp = -temp; }
        while(temp > 0) { rev[j++] = '0' + (temp % 10); temp /= 10; }
        while(j > 0) buf[i++] = rev[--j];
    }
    buf[i] = 0;
    uart_print(buf);
}

void uart_print_float(float value) {
    int int_part = (int)value;
    float frac = value - int_part;
    if(frac < 0) frac = -frac;
    int frac_int = (int)(frac * 100);
    uart_print_int(int_part);
    uart_print(".");
    if(frac_int < 10) uart_print("0");
    uart_print_int(frac_int);
}

void uart_println(const char *s) {
    uart_print(s);
    uart_print("\r\n");
}

/* LED control */
void leds_on(void) {
    HAL_GPIO_WritePin(LED1_PORT, LED1_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LED2_PORT, LED2_PIN, GPIO_PIN_SET);
}

void leds_off(void) {
    HAL_GPIO_WritePin(LED1_PORT, LED1_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED2_PORT, LED2_PIN, GPIO_PIN_RESET);
}

/* DSP functions */
void remove_dc_offset(float *data, int len) {
    float mean = 0;
    for(int i = 0; i < len; i++) mean += data[i];
    mean /= len;
    for(int i = 0; i < len; i++) data[i] -= mean;
}

void normalize_audio(float *data, int len) {
    float max_val = 0.0f;
    for(int i = 0; i < len; i++) {
        float a = fabsf(data[i]);
        if(a > max_val) max_val = a;
    }
    if(max_val < 1e-6f) return;
    for(int i = 0; i < len; i++) {
        data[i] /= max_val;
    }
}

float compute_energy(float *data, int len) {
    float energy = 0.0f;
    for (int i = 0; i < len; i++) {
        energy += data[i] * data[i];
    }
    return energy / len;
}

float compute_variance(float *data, int len) {
    float mean = 0.0f;
    for (int i = 0; i < len; i++) mean += data[i];
    mean /= len;

    float var = 0.0f;
    for (int i = 0; i < len; i++) {
        float d = data[i] - mean;
        var += d * d;
    }
    return var / len;
}

uint8_t has_energy(float *data, int len) {
    float energy = compute_energy(data, len);
    if (noise_floor_calibrated) {
        return energy > (noise_floor * NOISE_FLOOR_MULTIPLIER);
    } else {
        return energy > MIN_SPEECH_ENERGY;
    }
}

uint8_t has_variance(float *data, int len) {
    float variance = compute_variance(data, len);
    return variance > VARIANCE_THRESHOLD;
}

uint8_t vad_detect_webrtc(float *data, int len, int sample_rate, Fvad *vad) {
    const int frame_samples = sample_rate / 100;
    if (len % frame_samples != 0) {
        len = (len / frame_samples) * frame_samples;
    }

    int speech_frames = 0;
    int total_frames = len / frame_samples;

    for (int i = 0; i < len; i += frame_samples) {
        int16_t pcm[frame_samples];
        for (int j = 0; j < frame_samples; j++) {
            float scaled = data[i + j];
            if (scaled > 1.0f) scaled = 1.0f;
            if (scaled < -1.0f) scaled = -1.0f;
            int16_t sample = (int16_t)(scaled * 32767.0f);
            pcm[j] = sample;
        }
        int result = fvad_process(vad, pcm, frame_samples);
        if (result == 1) speech_frames++;
    }

    #ifdef DEBUG_VAD
    uart_print("VAD: "); uart_print_int(speech_frames);
    uart_print(" / "); uart_print_int(total_frames);
    uart_print("\r\n");
    #endif

    return (speech_frames > total_frames * VAD_FRACTION_REQUIRED);
}

/* Statistics functions (unchanged) */
float mean(const float* vec, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) sum += vec[i];
    return sum / len;
}

float std_dev(const float* vec, int len, float mean_val) {
    float sum_sq = 0;
    for (int i = 0; i < len; i++) {
        float d = vec[i] - mean_val;
        sum_sq += d * d;
    }
    return sqrtf(sum_sq / len);
}

float skewness(const float* vec, int len, float mean_val, float std) {
    float skew = 0;
    for (int i = 0; i < len; i++) {
        float z = (vec[i] - mean_val) / (std + 1e-6f);
        skew += z * z * z;
    }
    return skew / len;
}

float kurtosis(const float* vec, int len, float mean_val, float std) {
    float kurt = 0;
    for (int i = 0; i < len; i++) {
        float z = (vec[i] - mean_val) / (std + 1e-6f);
        kurt += z * z * z * z;
    }
    return kurt / len;
}

float first_moment_exact(const float* vec, int len) {
    float E = 0.0f, f1_sum = 0.0f;
    for (int i = 0; i < len; i++) {
        float x = vec[i];
        E += x;
        f1_sum += i * x;
    }
    if (E < 1e-6f) return 0.0f;
    return f1_sum / E;
}

float second_moment_safe(const float* vec, int len) {
    float E = 0.0f, f1_sum = 0.0f, f2_sum = 0.0f;
    for (int f = 0; f < len; f++) {
        float val = vec[f];
        E += val;
        f1_sum += f * val;
        f2_sum += f * f * val;
    }
    if (E < 1e-6f) return 0.0f;
    float f1 = f1_sum / E;
    float f2 = f2_sum / E;
    return (f2 - (f1 * f1)) / E;
}

float sparsity(const float* vec, int len) {
    float sum_abs = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_abs += fabsf(vec[i]);
        sum_sq  += vec[i] * vec[i];
    }
    if (sum_sq < 1e-6f) return 0.0f;
    float root_len = sqrtf((float)len);
    float ratio = (root_len * sum_abs) / sum_sq;
    return log10f(ratio - root_len + 1e-6f);
}

float discontinuity(const float* vec, int len) {
    float sum_diff_sq = 0.0f;
    for (int i = 0; i < len - 1; i++) {
        float d = vec[i + 1] - vec[i];
        sum_diff_sq += d * d;
    }
    return log10f(sum_diff_sq + 1e-6f);
}

void extract_all_stats(const float* data, int len, float* out_features) {
    float m = mean(data, len);
    float sd = std_dev(data, len, m);
    float f1 = first_moment_exact(data, len);
    out_features[0] = m;
    out_features[1] = sd;
    out_features[2] = skewness(data, len, m, sd);
    out_features[3] = kurtosis(data, len, m, sd);
    out_features[4] = f1;
    out_features[5] = second_moment_safe(data, len);
    out_features[6] = sparsity(data, len);
    out_features[7] = discontinuity(data, len);
}

/* ========================== DMA CALLBACKS ========================== */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s) {
    float *target = (active_buffer == 0) ? audio_ping : audio_pong;

    /* Process first half of buffer (indices 0 to DMA_BUF_SIZE/2 - 1) */
    for (int i = 0; i < DMA_BUF_SIZE / 2; i += 2) {
        if ((recording || calibrating) && buffer_index < CHUNK_SIZE) {
#ifdef DEBUG_RAW
            if (i < 8) {
                uint32_t combined = ((uint32_t)dma_buffer[i] << 16) | dma_buffer[i+1];
                uart_print("RAW="); uart_print_int((int)combined); uart_print("\r\n");
            }
#endif
            /* Reconstruct 24-bit sample from two consecutive 16-bit half-words */
            uint32_t hi = dma_buffer[i];
            uint32_t lo = dma_buffer[i+1];
            int32_t raw = (int32_t)((hi << 16) | lo);
            raw >>= 8;   /* now raw is a 24-bit signed integer */
            float sample = (float)raw * (1.0f / 8388608.0f);   /* scale to [-1,1] */

            if (!clipping_reported && (sample > 1.0f || sample < -1.0f)) {
                clipping_reported = 1;
                uart_println("Warning: Audio clipping detected!");
            }

            /* DC blocking filter (high-pass ~ 5 Hz) */
            float y = sample - x_prev + 0.995f * y_prev;
            x_prev = sample;
            y_prev = y;

            target[buffer_index++] = y;

            if (buffer_index >= CHUNK_SIZE) {
                chunk_ready = 1;
                active_buffer ^= 1;
                buffer_index = 0;
                x_prev = 0.0f;
                y_prev = 0.0f;
            }
        }
    }
}

void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s) {
    float *target = (active_buffer == 0) ? audio_ping : audio_pong;

    /* Process second half of buffer (indices DMA_BUF_SIZE/2 to DMA_BUF_SIZE-1) */
    for (int i = DMA_BUF_SIZE / 2; i < DMA_BUF_SIZE; i += 2) {
        if ((recording || calibrating) && buffer_index < CHUNK_SIZE) {
#ifdef DEBUG_RAW
            if (i < DMA_BUF_SIZE/2 + 8) {
                uint32_t combined = ((uint32_t)dma_buffer[i] << 16) | dma_buffer[i+1];
                uart_print("RAW="); uart_print_int((int)combined); uart_print("\r\n");
            }
#endif
            uint32_t hi = dma_buffer[i];
            uint32_t lo = dma_buffer[i+1];
            int32_t raw = (int32_t)((hi << 16) | lo);
            raw >>= 8;
            float sample = (float)raw * (1.0f / 8388608.0f);

            if (!clipping_reported && (sample > 1.0f || sample < -1.0f)) {
                clipping_reported = 1;
                uart_println("Warning: Audio clipping detected!");
            }

            float y = sample - x_prev + 0.995f * y_prev;
            x_prev = sample;
            y_prev = y;

            target[buffer_index++] = y;

            if (buffer_index >= CHUNK_SIZE) {
                chunk_ready = 1;
                active_buffer ^= 1;
                buffer_index = 0;
                x_prev = 0.0f;
                y_prev = 0.0f;
            }
        }
    }
}

/* Calibrate noise floor */
void calibrate_noise_floor(void) {
    uart_println("Calibrating noise floor... Stay silent!");
    calibrating = 1;

    float total_energy = 0.0f;
    for (int c = 0; c < CALIBRATION_CHUNKS; c++) {
        while (!chunk_ready) { __NOP(); }
        chunk_ready = 0;

        float *buffer = (active_buffer == 0) ? audio_pong : audio_ping;
        total_energy += compute_energy(buffer, CHUNK_SIZE);
    }

    noise_floor = total_energy / CALIBRATION_CHUNKS;
    noise_floor_calibrated = 1;
    calibrating = 0;

    active_buffer = 0;
    buffer_index = 0;
    chunk_ready = 0;
    memset(audio_ping, 0, sizeof(audio_ping));
    memset(audio_pong, 0, sizeof(audio_pong));

    uart_print("Noise floor = ");
    uart_print_float(noise_floor);
    uart_print("\r\n");
    uart_println("Calibration complete. You can now press the button.");
}

/* Process a single chunk – prints everything */
void process_chunk(float *buffer, int chunk_num) {
    uart_print("\r\n===== CHUNK =====\r\n");

    float max_val = 0.0f;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        float a = fabsf(buffer[i]);
        if (a > max_val) max_val = a;
    }
    uart_print("Max amplitude: ");
    uart_print_float(max_val);
    uart_print("\r\n");

    float energy = compute_energy(buffer, CHUNK_SIZE);
    float delta = energy - noise_floor;
    uart_print("Energy: "); uart_print_float(energy); uart_print("\r\n");
    uart_print("Delta: "); uart_print_float(delta); uart_print("\r\n");

    uint8_t vad_result = vad_detect_webrtc(buffer, CHUNK_SIZE, 16000, vad_inst);
    uart_print("VAD result: "); uart_print_int(vad_result); uart_print("\r\n");

    uart_print("PROCESSING (TEST MODE)\r\n");

    static float gfcc_matrix[MAX_FRAMES][GFCC_NUM_COEFFS];
    static float mfcc_matrix[MAX_FRAMES][13];
    uint32_t num_frames_actual = 0;

    for(uint32_t start=0, f_idx=0;
        start+FRAME_SIZE<=CHUNK_SIZE;
        start+=FRAME_HOP, f_idx++) {

        // GFCC (pre‑emphasized)
        for(int i=0;i<FRAME_SIZE;i++) {
            float x = buffer[start+i];
            if(i>0) frame_buffer[i] = x - 0.97f * buffer[start+i-1];
            else frame_buffer[i] = x;
        }
        GFCC_Process(&gfcc, frame_buffer, features);
        for(int k=0;k<GFCC_NUM_COEFFS;k++) {
            gfcc_matrix[f_idx][k] = features[k];
        }

        // MFCC
        float mfcc_coeffs[13];
        compute_mfcc(&buffer[start], FRAME_SIZE, mfcc_coeffs);
        for(int k=0;k<13;k++) {
            mfcc_matrix[f_idx][k] = mfcc_coeffs[k];
        }

        num_frames_actual = f_idx + 1;
    }

    uart_print("\r\nGFCC MATRIX (frames x coefficients)\r\n");
    for(uint32_t f=0; f<num_frames_actual; f++) {
        for(int k=0;k<GFCC_NUM_COEFFS;k++) {
            uart_print_float(gfcc_matrix[f][k]);
            if(k < GFCC_NUM_COEFFS-1) uart_print(",");
        }
        uart_print("\r\n");
    }

    uart_print("\r\nMFCC MATRIX (frames x coefficients)\r\n");
    for(uint32_t f=0; f<num_frames_actual; f++) {
        for(int k=0;k<13;k++) {
            uart_print_float(mfcc_matrix[f][k]);
            if(k < 12) uart_print(",");
        }
        uart_print("\r\n");
    }

    compute_dynamic_spectrogram(buffer, CHUNK_SIZE, FRAME_SIZE, FRAME_HOP, FFT_SIZE,
                                (float(*)[MAX_BINS])spectrogram, &num_frames);

    uart_print("\r\nNNMF W (frequency basis)\r\n");
    nnmf_rank1_iter((float(*)[MAX_BINS])spectrogram, W, H, NNMF_ITER);
    for(uint32_t f=0; f<MAX_BINS; f++) {
        uart_print_float(W[f]);
        if(f<MAX_BINS-1) uart_print(",");
    }
    uart_print("\r\n");

    uart_print("\r\nNNMF H (temporal activation)\r\n");
    for(uint32_t t=0; t<num_frames; t++) {
        uart_print_float(H[t]);
        if(t<num_frames-1) uart_print(",");
    }
    uart_print("\r\n");

    extract_all_stats(W, MAX_BINS, stats_w);
    extract_all_stats(H, num_frames, stats_h);

    uart_print("\r\nW STATS (mean, std, skewness, kurtosis, 1st moment, 2nd moment, sparsity, discontinuity)\r\n");
    for(int i=0;i<8;i++) {
        uart_print_float(stats_w[i]);
        if(i<7) uart_print(",");
    }
    uart_print("\r\n");

    uart_print("\r\nH STATS (same order)\r\n");
    for(int i=0;i<8;i++) {
        uart_print_float(stats_h[i]);
        if(i<7) uart_print(",");
    }
    uart_print("\r\n");
}

/* MAIN */
int main(void) {
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_DMA_Init();
    MX_USART2_UART_Init();
    MX_I2S2_Init();

    HAL_Delay(1000);

    GFCC_Init(&gfcc);

    vad_inst = fvad_new();
    if (!vad_inst) {
        uart_println("VAD init failed!");
        while(1);
    }
    fvad_set_sample_rate(vad_inst, 16000);
    fvad_set_mode(vad_inst, 3);

    leds_off();
    uart_println("STREAMING ALS READY (Circular DMA)");

    HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)dma_buffer, DMA_BUF_SIZE);

    calibrate_noise_floor();

    uart_println("Press and release PA0 to record 3 seconds");

    while(1) {
        if(system_locked) continue;

        uint8_t button = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0);

        if(prev_button && !button && (HAL_GetTick() - last_press_time > 300)) {
            last_press_time = HAL_GetTick();
            button_released = 1;
        }
        prev_button = button;

        if(button_released && !recording && !system_locked) {
            button_released = 0;
            recording = 1;
            system_locked = 0;

            active_buffer = 0;
            buffer_index = 0;
            chunk_ready = 0;
            chunk_count = 0;
            x_prev = 0.0f;
            y_prev = 0.0f;

            memset(audio_ping, 0, sizeof(audio_ping));
            memset(audio_pong, 0, sizeof(audio_pong));

            leds_on();
            uart_println("\r\n=== RECORDING START ===");
        }

        if(recording && chunk_ready) {
            chunk_ready = 0;
            float *buffer = (active_buffer == 0) ? audio_pong : audio_ping;
            process_chunk(buffer, chunk_count + 1);
            chunk_count++;

            if(chunk_count >= TOTAL_CHUNKS) {
                uart_println("\r\n=== 3 SEC DONE ===");
                recording = 0;
                system_locked = 1;
                leds_off();
            }
        }
    }
}
/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2S2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2S2_Init(void)
{

  /* USER CODE BEGIN I2S2_Init 0 */

  /* USER CODE END I2S2_Init 0 */

  /* USER CODE BEGIN I2S2_Init 1 */

  /* USER CODE END I2S2_Init 1 */
  hi2s2.Instance = SPI2;
  hi2s2.Init.Mode = I2S_MODE_MASTER_RX;
  hi2s2.Init.Standard = I2S_STANDARD_PHILIPS;
  hi2s2.Init.DataFormat = I2S_DATAFORMAT_24B;
  hi2s2.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE;
  hi2s2.Init.AudioFreq = I2S_AUDIOFREQ_16K;
  hi2s2.Init.CPOL = I2S_CPOL_HIGH;
  hi2s2.Init.ClockSource = I2S_CLOCK_PLL;
  hi2s2.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
  if (HAL_I2S_Init(&hi2s2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2S2_Init 2 */

  /* USER CODE END I2S2_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6|GPIO_PIN_7, GPIO_PIN_RESET);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PA6 PA7 */
  GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
