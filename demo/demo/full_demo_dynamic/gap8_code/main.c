#include "modelKernels.h"
#include "Gap.h"
#include "model.h"
#include "bsp/display/ili9341.h" 
#include "pmsis.h"
static struct pi_device ili;
static pi_buffer_t buffer;

//#include "bsp/bsp.h"
//#include "bsp/camera.h"

#include <stdio.h>
#include "rt/rt_api.h"
//#include "rt/rt_cluster.h"
#include <stdint.h>

#define imgW    324
#define imgH    244
#define CROP    40

#define CAM_WIDTH    324
#define CAM_HEIGHT   244

#define LCD_WIDTH    320
#define LCD_HEIGHT   240

#define OFFSET imgW*imgH

#define BAUD 3000000

//Sizes for testing
//#define imgW    20
//#define imgH    10


#define DEBUG_PRINTF printf

#define STACK_SIZE     1024
#define SLAVE_STACK_SIZE 1024
#define RX_BUFFER_SIZE 1
#define BUFFER_SIZE 1

#define CAMERA_WIDTH  324
#define CAMERA_HEIGHT 244

#define INPUT_WIDTH 244
#define INPUT_HEIGHT 244
#define INPUT_CHANNEL 1

#define OUTPUT_WIDTH 30
#define OUTPUT_HEIGHT 30
#define OUTPUT_CHANNEL 8

#define OUTPUT_SIZE OUTPUT_CHANNEL*OUTPUT_WIDTH*OUTPUT_HEIGHT

AT_HYPERFLASH_FS_EXT_ADDR_TYPE model_L3_Flash = 0;

//L2_MEM short int *ResOut;
//L2_MEM short int *ImgIn;
//L2_MEM short int *ImgIn;
//L2_MEM uint8_t *imgBuffer;


RT_FC_TINY_DATA rt_camera_t *camera1;
RT_FC_TINY_DATA rt_cam_conf_t *cam1_conf;
RT_FC_TINY_DATA unsigned int  err = 0;


//RT_L2_DATA uint8_t imgBuff0[imgW*imgH*2];
//RT_L2_DATA uint8_t imgBuff0[imgW*imgH*2];
RT_L2_DATA uint8_t imgBuff0[CAMERA_HEIGHT*CAMERA_WIDTH*2];
//RT_L2_DATA short int other_buff[INPUT_HEIGHT*INPUT_WIDTH*2];
int16_t* imgBuff0_short = (int16_t *) imgBuff0;

RT_L2_DATA short int ResOut[OUTPUT_SIZE];

//array to hold timing
//times[0] = camera capture time
//times[1] = nn forward pass time
//times[2] = total time for uart writes (both image and activations)
//measured in microseconds so store using 4 byte unsigned ints 
RT_L2_DATA uint32_t times[3];

//RT_L2_DATA uint8_t rx_buffer[2];

//static struct pi_device uart;
//L2_MEM int8_t *uartBuffer;

struct pi_device uart;
struct pi_uart_conf uart_conf;

static uint8_t *rx_buffer;


void in_place_byte_to_short_and_crop(uint8_t *buff, int16_t *buff_short, int H, int W, int crop){
	int cropW = W - 2*crop;
    //short int n = 128;
	for (int i=0; i<H; i++){
		for (int j=0; j<cropW; j++){
			buff_short[i*(cropW) + j] = buff[i*imgW + j+crop];
            //buff_short[i*(cropW) + j] = (buff_short[i*(cropW) + j] - n) * n;
		}
	}	
}

void write_cropped_bytes(uint8_t *buff, int H, int W, int crop){
	int cropW = W - 2*crop;
	for (int i=0; i<H; i++) {
		pi_uart_write(&uart, buff + i*imgW + crop, 244);
	}	
}


static void cam_param_conf_perso(rt_cam_conf_t *conf){
  conf->resolution = QVGA;
  conf->format = HIMAX_MONO_COLOR;
  conf->fps = fps30;
  conf->slice_en = DISABLE;
  conf->shift = 0;
  conf->frameDrop_en = DISABLE;
  conf->frameDrop_value = 0;
  conf->cpiCfg = UDMA_CHANNEL_CFG_SIZE_16;
  conf->control_id = 1;
}


static void forward() {
    //modelCNN(imgBuff0_short, imgBuff0_short);
    modelCNN(imgBuff0_short, ResOut);
}


void uart_read() {
	pi_pad_set_function(PI_PAD_46_B7_SPIM0_SCK, PI_PAD_FUNC0);
    rx_buffer = (uint8_t *) pmsis_l2_malloc((uint32_t) 2);
    pi_task_t wait_task = {0};
    pi_task_block(&wait_task);
    //printf("tasK_block passed\n");
    pi_uart_read_async(&uart, rx_buffer, 2, &wait_task);
    //printf("pi_uart_async_read called, waiting\n");
    pi_task_wait_on(&wait_task);
    //printf("waiting over, got 1 byte with value %d\n",  rx_buffer[0]);
	pi_pad_set_function(PI_PAD_46_B7_SPIM0_SCK, PI_PAD_FUNC3);
}




void run(void) {
    printf("Entering Main Controller\n");
    //rt_freq_set(RT_FREQ_DOMAIN_FC,250000000);
    //rt_freq_set(RT_FREQ_DOMAIN_CL, 175000000);

    if (rt_event_alloc(NULL, 1)) printf("is null\n");
    rt_event_t *event = rt_event_get_blocking(NULL);
    
    rt_cluster_mount(1, 0, NULL, NULL);

    // Initialize the parameter of camera.
    cam1_conf = rt_alloc(RT_ALLOC_FC_DATA, sizeof(rt_cam_conf_t));
    rt_camera_conf_init(cam1_conf);   // Init the camera
    cam_param_conf_perso(cam1_conf);
    // open the camera, the conf structure will be stored in camera object
    camera1 = rt_camera_open("camera", cam1_conf, 0);
    if (camera1 == NULL) printf("camera is null\n");
    // free the memory of camera configuration structure
    rt_free(RT_ALLOC_FC_DATA, cam1_conf, sizeof(rt_cam_conf_t));

    //Init Camera Interface
    rt_cam_control(camera1, CMD_INIT, 0);
    rt_time_wait_us(1000000); //Wait camera calibration
    printf("camera init finished\n");

    struct pi_ili9341_conf ili_conf;
    pi_ili9341_conf_init(&ili_conf);
    pi_open_from_conf(&ili, &ili_conf);
    pi_display_open(&ili);
    pi_display_ioctl(&ili, PI_ILI_IOCTL_ORIENTATION, (void *)PI_ILI_ORIENTATION_90);
    buffer.data = imgBuff0 +OFFSET +CAM_WIDTH*2+2;
    buffer.stride = 4;
    //WIth Himax, propertly configure the buffer to skip boarder pixels
    pi_buffer_init(&buffer, PI_BUFFER_TYPE_L2, imgBuff0+OFFSET+CAM_WIDTH*2+2);
    pi_buffer_set_stride(&buffer, 4);
    pi_buffer_set_format(&buffer, CAM_WIDTH, CAM_HEIGHT, 1, PI_BUFFER_FORMAT_GRAY);	
    printf("display init finished\n");
    
    int ret = modelCNN_Construct();
    if (ret != 0) {
        printf("[ERROR] CNN construct returned %d\n", ret);
    } else {
        printf("model constructed\n");
    }
    
    //ResOut = (short int *) AT_L2_ALLOC(0, 32*30*30*sizeof(short int));
    //if (ResOut == 0) {
    //    printf("[ERROR] failed to alloc for output\n");
    //} else {
    //    printf("output memory alloc success\n");
    //}
	
	
    pi_uart_conf_init(&uart_conf);
    uart_conf.baudrate_bps = BAUD;
    uart_conf.uart_id = 0;
    uart_conf.enable_tx = 1;
    uart_conf.enable_rx = 1;

    pi_open_from_conf(&uart, &uart_conf);
    if (pi_uart_open(&uart)) {
        printf("uart open failed\n");
    } else {
    	printf("uart open\n");
    }

    printf("entered main loop\n");
    uint32_t start;
    while (1) {
        times[2] = 0; //location for uart timing
		
        //Wait for control signal
        //reads 2 bytes into the rx_buffer
        //rx_buffer[0] = boolean to determine if we send back image pixels
        //rx_buffer[1] = number of channels of activations to send back
	    uart_read();
        
		
		//Capture image
        start = rt_time_get_us();
        rt_cam_control(camera1, CMD_START, 0);
        rt_event_t *event = rt_event_get_blocking(NULL);
        rt_camera_capture (camera1, imgBuff0 + OFFSET, imgW*imgH, event);
        rt_event_wait(event);
        rt_cam_control(camera1, CMD_PAUSE, 0);
        times[0] = rt_time_get_us() - start;
        //printf("image captured\n");
	    
        //display image on LCD    
		//pi_display_write(&ili, &buffer, 0, 0, LCD_WIDTH, LCD_HEIGHT);

		
		//Set horizontal crop amount
		//Each side is cropped by CROP amount
        if (rx_buffer[0]) {
            start = rt_time_get_us();
            write_cropped_bytes(imgBuff0+imgW*imgH, imgH, imgW, CROP);
            times[2] += (rt_time_get_us() - start);
        }


        start = rt_time_get_us(); //start timing NN forward pass
		
        //Convert to array of shorts in-place
		in_place_byte_to_short_and_crop(imgBuff0+imgW*imgH, imgBuff0_short, imgH, imgW, CROP);
        
        //normalize NN inputs 
        short int n = 128;
        for (int i = 0; i < 244*244; i++) {
            imgBuff0_short[i] = (imgBuff0_short[i] - n) * n;
        }

        //network forward pass
        rt_cluster_call(NULL, 0, (void *) &forward, NULL, NULL, STACK_SIZE, STACK_SIZE, 0, NULL);
        //printf("forward pass complete\n");
        times[1] = rt_time_get_us() - start;
        
        //write the activations out over UART one channel at a time
        start = rt_time_get_us();
        for (int i = 0; i < rx_buffer[1]; i++) {
            pi_uart_write(&uart, ResOut + i*30*30, 30*30*2);
        }
        times[2] += (rt_time_get_us() - start);

        //write out time measurements
        pi_uart_write(&uart, times, 3*sizeof(uint32_t));
    } //main loop end
    modelCNN_Destruct();
}

int main() {
    return pmsis_kickoff((void *) run);
}
