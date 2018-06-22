#include "mbed.h"

#include "TCPSocket.h"
#include "EthernetInterface.h"
#include "SocketAddress.h"

//#include "kws_ds_cnn.h"
#include "kws_cnn.h"

#include "mbed_stats.h"
extern mbed_stats_heap_t *pStaticHeapStats;
void mbed_stats_heap_reset_max_size()
{
    pStaticHeapStats->max_size = pStaticHeapStats->current_size;
}
// #include "mbed_mem_trace.h"


// Network interface to pc
EthernetInterface eth;
TCPSocket socket;
TCPServer tcpserver;
#define IP         "192.168.0.211"
#define GATEWAY    "192.168.0.210"
#define MASK       "255.255.255.0"

// Buffer for audio and communication to pc
const int audio_data_length = 16000;
int16_t audio_data[audio_data_length] = {0};
const int COM_BUFFER_SIZE = 4096;
uint8_t COM_BUFFER[COM_BUFFER_SIZE];

int idx_max_class = -1;  // classes: {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"}
int prediction_confidence = -1;  // in %

// Measure execution time
Timer T;
int T_ms_features_extraction = -1;
int T_ms_inference = -1;

// Measure memory statistics
const int nb_memory_stats = 4;
// Index refering to part of code execution
int code_part = -1;

unsigned long int Heap_stats[nb_memory_stats] = {0};
unsigned long int Stack_stats[nb_memory_stats] = {0};
// unsigned long int dynamic_allocated_memory[nb_memory_stats] = {0};


// If needed, send char indicating normal execution on mbed side. 
// Then exepects client to close socket, and closes socket on mbed side.
int close_socket_on_normal_exec(bool send_flag)
{
    if (send_flag)
    {
        char response = '0';
        socket.send(&response, 1);
    }
    // Client should then close the socket
    int client_response = socket.recv(COM_BUFFER, COM_BUFFER_SIZE);
    if (client_response == 0)
    {
        socket.close();
    }        
    else
        NVIC_SystemReset();
        // return client_response; // Something wrong happened.
    return 0;
}

int read_audio()
{
    // Receive audio from client and store it in audio_buffer.
    int sample_index = 0;
    int16_t byte_not_read;
    bool one_byte_left = false;
    while(sample_index < audio_data_length)
    {
        int num_bytes_received = socket.recv(COM_BUFFER, COM_BUFFER_SIZE);

        if (one_byte_left == true)
        {
            audio_data[sample_index] = (int16_t) ((COM_BUFFER[0] << 8) | byte_not_read);
            sample_index++;
        }

        
        for (int i = one_byte_left ? 1 : 0; i+1 < num_bytes_received; i+=2)
        {
            audio_data[sample_index] = (int16_t) ((COM_BUFFER[i+1] << 8) | COM_BUFFER[i]);
            sample_index++;
        }

        one_byte_left = (num_bytes_received % 2 == 1 && !one_byte_left) || (num_bytes_received % 2 == 0 && one_byte_left);

        if (one_byte_left % 2 == 1)
            byte_not_read = COM_BUFFER[num_bytes_received -1];
    }

    // we have read all audio data. Send response to client
    int closed = close_socket_on_normal_exec(true);
    // Return number of integer16 read, or return value from socket.close()
    return (closed == 0) ? sample_index : closed;
}

int send_audio()
{
    int n_bytes_sent = 0;
    while (n_bytes_sent < 2*audio_data_length)
    {
        n_bytes_sent += socket.send((char *)(audio_data) + n_bytes_sent, audio_data_length);
    }
    return close_socket_on_normal_exec(false);
}

/*
void mem_trace_callback(uint8_t op, void *res, void *caller, ...) {
    va_list va;
    size_t temp_s1, temp_s2;
    void *temp_ptr;

    printf("In callback !\r\n");
    if (code_part < 0 || code_part > nb_memory_stats)
        return;
    
    va_start(va, caller);
    switch(op) {
        case MBED_MEM_TRACE_MALLOC:
            temp_s1 = va_arg(va, size_t);
            printf(MBED_MEM_DEFAULT_TRACER_PREFIX "m:%p;%p-%u\r\n", res, caller, temp_s1);
            dynamic_allocated_memory[code_part] += temp_s1;
            break;

        case MBED_MEM_TRACE_REALLOC:
            temp_ptr = va_arg(va, void*);
            temp_s1 = va_arg(va, size_t);
            printf(MBED_MEM_DEFAULT_TRACER_PREFIX "r:%p;%p-%p;%u\r\n", res, caller, temp_ptr, temp_s1);
            break;

        case MBED_MEM_TRACE_CALLOC:
            temp_s1 = va_arg(va, size_t);
            temp_s2 = va_arg(va, size_t);
            printf(MBED_MEM_DEFAULT_TRACER_PREFIX "c:%p;%p-%u;%u\r\n", res, caller, temp_s1, temp_s2);
            dynamic_allocated_memory[code_part] += temp_s1 * temp_s2;
            break;

        case MBED_MEM_TRACE_FREE:
            temp_ptr = va_arg(va, void*);
            printf(MBED_MEM_DEFAULT_TRACER_PREFIX "f:%p;%p-%p\r\n", res, caller, temp_ptr);
            break;

        default:
            printf("?\r\n");
    }
    va_end(va);
}
*/

int inference()
{
    mbed_stats_heap_t Hstats;
    mbed_stats_stack_t Sstats;
    code_part = 0;
    mbed_stats_heap_get(&Hstats); Heap_stats[code_part] = Hstats.max_size; // mbed_stats_heap_reset_max_size();
    mbed_stats_stack_get(&Sstats); Stack_stats[code_part] = Sstats.max_size;

    // KWS_DS_CNN kws(audio_data);
    KWS_CNN kws(audio_data);
    code_part = 1;
    /*void *allocation = malloc(10000);
    {int blob[1000];}
    free(allocation);*/
    
    mbed_stats_heap_get(&Hstats); Heap_stats[code_part] = Hstats.max_size; // mbed_stats_heap_reset_max_size();
    mbed_stats_stack_get(&Sstats); Stack_stats[code_part] = Sstats.max_size;

    int start;
    // extract mfcc features
    T.reset(); T.start(); start = T.read_ms();
    code_part = 2;
    kws.extract_features();
    T_ms_features_extraction = T.read_ms() - start; T.stop();
    mbed_stats_heap_get(&Hstats); Heap_stats[code_part] = Hstats.max_size; // mbed_stats_heap_reset_max_size();
    mbed_stats_stack_get(&Sstats); Stack_stats[code_part] = Sstats.max_size;

    // classify using network
    T.reset(); T.start(); start = T.read_ms();
    code_part = 3;
    kws.classify();
    T_ms_inference = T.read_ms() - start; T.stop();
    mbed_stats_heap_get(&Hstats); Heap_stats[code_part] = Hstats.max_size; // mbed_stats_heap_reset_max_size();
    mbed_stats_stack_get(&Sstats); Stack_stats[code_part] = Sstats.max_size;
    
    code_part = -1;
    idx_max_class = kws.get_top_class(kws.output);
    prediction_confidence = ((int)kws.output[idx_max_class]*100/128);

    return close_socket_on_normal_exec(true);  // return either 0 (normal exit) or return value from socket.close()
}

int send_predictions()
{
    socket.send(&idx_max_class, sizeof(idx_max_class));
    socket.send(&prediction_confidence, sizeof(prediction_confidence));
    idx_max_class = -1;
    prediction_confidence = -1;
    return close_socket_on_normal_exec(false);  // return either 0 (normal exit) or return value from socket.close()
}

int send_time_stats()
{   
    socket.send(&T_ms_features_extraction, sizeof(T_ms_features_extraction));
    socket.send(&T_ms_inference, sizeof(T_ms_features_extraction));
    T_ms_features_extraction = -1;
    T_ms_inference = -1;
    return close_socket_on_normal_exec(false);  // return either 0 (normal exit) or return value from socket.close()
}

int send_memory_stats()
{
    socket.send(Heap_stats, sizeof(Heap_stats));
    socket.send(Stack_stats, sizeof(Stack_stats));
    // socket.send(dynamic_allocated_memory, sizeof(dynamic_allocated_memory));

    for (int i = 0; i < nb_memory_stats; ++i)
    {
        Heap_stats[i] = 0;
        Stack_stats[i] = 0;
        // dynamic_allocated_memory[i] = 0;
    }
    
    return close_socket_on_normal_exec(false);
}

void handle_connection()
{
    char request_type = '0';
    socket.recv(&request_type, 1);
    // printf("Command %c\r\n", request_type);
    switch (request_type)
    {
        case 'r':
        // receive audio data from client and send adequate response
            read_audio();
            break;
        case 's':
        // Send the audio to pc
            send_audio();
            break;
        case 'i':
        // run inference of the network on current audio data and send response
            inference();
            break;
        case 'p':
        // Client asks for predicted class and confidence
            send_predictions();
            break;
        case 't':
        // Client asks for time measurements of last inference
            send_time_stats();
            break;
        case 'm':
        // Client asks for memory measurments of last inference
            send_memory_stats();
            break;
        default:
        // Unrecognized request identifier: send error to client
            const char error[26] = "Unknown request received";
            socket.send(error, 26);
            break;
    }
}

void get_client_connection()
{
    while (true)
    {
        if (tcpserver.accept(&socket) < 0)
        {
            printf("Warning: failed to accept connection.\n\r");
            continue;
        }
        handle_connection();
    }
}

int main()
{
    // mbed_mem_trace_set_callback(mem_trace_callback);

    printf("Network-socket TCP Server benchmarking !\r\n");
    eth.disconnect();
    int i=eth.set_network(IP,MASK,GATEWAY);
    printf("set IP status: %i \r\n",i);
    i=eth.connect();
    printf("connect status: %i \r\n",i);
    const char *ip = eth.get_ip_address();
    const char *mac = eth.get_mac_address();
    printf("IP address is: %s\n\r", ip ? ip : "No IP");
    printf("MAC address is: %s\n\r", mac ? mac : "No MAC");
    SocketAddress sockaddr;
    i=tcpserver.open(&eth);
    if (i<0) 
    {
        printf("open error: %i \r\n",i);
        while(true) {}
    }
    printf("server open\r\n");
    // i=tcpserver.bind(ip,23);
    i=tcpserver.bind(5005);
    if (i<0)
    {
        printf("bind error: %i \r\n",i);
        while(true) {}
    }
    printf("Bound\r\n");

    if (tcpserver.listen()<0) printf("listen error\r\n");
    printf("Listening\r\n");

    get_client_connection();
}
