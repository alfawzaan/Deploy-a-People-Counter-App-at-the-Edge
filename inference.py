#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_path, device="CPU", cpu_extension = None):
        model_weights = model_path[:-3]+"bin"
        self.plugin = IECore()
        self.network = IENetwork(model = model_path, weights = model_weights)
        
        ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
          ### TODO: Load the model ###
        self.exec_network = self.plugin.load_network(self.network, device)
            
        # Check Network layer support
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for the specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
            log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
            sys.exit(1)

       
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        ### TODO: Return the loaded inference plugin ###
        return self.exec_network
       
        ### TODO: Check for supported layers ###
       
        
        ### Note: You may need to update the function parameters. ###
#         return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shape = self.network.inputs[self.input_blob].shape
#         print("LENGTH {}\n".format(input_shape))
        return input_shape

    def exec_net(self, inf_input):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        result = self.exec_network.start_async(request_id=0, inputs={self.input_blob:inf_input})
        return result

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
