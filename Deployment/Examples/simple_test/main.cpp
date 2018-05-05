/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description: Example code for running keyword spotting on Cortex-M boards
 */

#include "kws_ds_cnn.h"
#include "mbed_rpc.h"
// #include "wav_data.h"

Timer T;
Serial pc(USBTX, USBRX);

void inference(Arguments *input, Reply *output);
RPCFunction rpc_inference(&inference, "inference");


void inference(Arguments *input, Reply *output)
{
  /*
  // Arguments are already parsed into argv array of char*
  printf("Object name = %s\r\n",input->obj_name);
  printf("Method name = %s\r\n",input->method_name);
  for (int i=0; i < input->argc; i++)
    printf("argv[%1d] = %s \r\n",i,input->argv[i]);
  */
  
  int16_t audio_buf[16000];
  for (int i=0; i < input->argc; ++i)
    audio_buf[i] = (int16_t)input->getArg<int>(); // convert to int and pray it is in range of short int
  
  char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};
  KWS_DS_CNN kws(audio_buf);

  T.start();
  int start=T.read_us();
  kws.extract_features(); //extract mfcc features
  kws.classify();	  //classify using dnn
  int end=T.read_us();
  T.stop();
  int max_ind = kws.get_top_class(kws.output);
  pc.printf("Total time : %d us\r\n",end-start);
  pc.printf("Detected %s (%d%%)\r\n",output_class[max_ind],((int)kws.output[max_ind]*100/128));
  
  char buffer[2000];
  sprintf(buffer, "Total time : %d us\r\n",end-start);
  output->putData(buffer);
  sprintf(buffer, "Detected %s (%d%%)\r\n",output_class[max_ind],((int)kws.output[max_ind]*100/128));
  output->putData(buffer);
}

int main()
{
  char buf[4000], outbuf[4000];
  while(1)
  {
    pc.scanf("%1999[^\r\n]", buf);
    RPC::call(buf, outbuf);
    pc.printf("%s\r\n", outbuf);
  }

  return 0;
}


/*

#include "kws_ds_cnn.h"
#include "wav_data.h"

int16_t audio_buffer[16000]=WAVE_DATA;

Timer T;
Serial pc(USBTX, USBRX);

int main()
{
  char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};
  KWS_DS_CNN kws(audio_buffer);

  T.start();
  int start=T.read_us();
  kws.extract_features(); //extract mfcc features
  kws.classify();	  //classify using dnn
  int end=T.read_us();
  T.stop();
  int max_ind = kws.get_top_class(kws.output);
  pc.printf("Total time : %d us\r\n",end-start);
  printf("Detected %s (%d%%)\r\n",output_class[max_ind],((int)kws.output[max_ind]*100/128));

  return 0;
}


*/