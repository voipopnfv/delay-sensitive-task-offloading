# coding: utf-8

import grpc

import service.objDetectMod_pb2 as objDetectMod_pb2
import service.objDetectMod_pb2_grpc as objDetectMod_pb2_grpc

def runInit():
    channel = grpc.insecure_channel('0.0.0.0:50052')
    stub = objDetectMod_pb2_grpc.objDetectModserviceStub(channel)
    request = objDetectMod_pb2.initArg(
        label_path = "label_class.pbtxt",
        num_classes = 5,
        queue_size = 5
        # gpudev = "0"
    )
    responce = stub.init(request)
    print(responce.status)

runInit()
print("The model is initialized.")
