"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    infer_network.load_model(args.model, args.device,  args.cpu_extension)
    n,c,h,w = infer_network.get_input_shape() #n, c,
    total_people = 0
    current_count = 0
    prev_count = 0
    duration = 0
    new_count = False
    if args.input == "CAM":
        args.input=0
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))
    frame_wait_duration = 0
    start_time = 0
    
    while cap.isOpened():
        flag, frame = cap.read()
        
        if flag== False:
            break
        key_pressed = cv2.waitKey(60)
        
        pre_pro_frame = cv2.resize(frame, (w, h))
        pre_pro_frame = pre_pro_frame.transpose((2,0,1))
        pre_pro_frame = pre_pro_frame.reshape((n,c,h, w))#n, c,
        inf_start = time.time()
        
        infer_network.exec_net(pre_pro_frame)
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start
            result = infer_network.get_output()
            current_count = 0
            for boxes in result[0][0]:
                if boxes[2] >= prob_threshold :
                    xmin = int(boxes[3]*width)
                    ymin = int(boxes[4]*height)
                    xmax = int(boxes[5]*width)
                    ymax = int(boxes[6]*height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(256, 256, 256), 1)
                    frame_wait_duration +=1   
                    current_count += 1
            inf_time_message = "Inference time: {:.3f}ms" \
                .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            if current_count > prev_count and frame_wait_duration%5==0:#and frame_wait_duration == 5
                start_time = time.time()
                total_people += current_count - prev_count
                client.publish("person", json.dumps({"total":total_people}))
    
            if current_count < prev_count and frame_wait_duration%5==0:
                duration = int(time.time()-start_time)
                client.publish("person/duration",json.dumps({"duration":duration}))
                
            
                
            client.publish("person",json.dumps({"count":current_count}))
            prev_count = current_count
            if key_pressed == 27:
                break
        
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
                                                    

    ### TODO: Load the model through `infer_network` ###

    ### TODO: Handle the input stream ###

    ### TODO: Loop until stream is over ###

        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###

        ### TODO: Start asynchronous inference for specified request ###

        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
