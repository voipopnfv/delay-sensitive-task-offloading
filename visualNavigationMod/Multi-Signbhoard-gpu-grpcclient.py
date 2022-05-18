# coding: utf-8

import grpc

import service.visualNavigationMod_pb2 as visualNavigationMod_pb2
import service.visualNavigationMod_pb2_grpc as visualNavigationMod_pb2_grpc

def runInit():
    channel = grpc.insecure_channel('localhost:50051')
    stub = visualNavigationMod_pb2_grpc.visualNavigationModserviceStub(channel)
    request = visualNavigationMod_pb2.initArg(
        input = "input/Cut2-030-U3_01_20190817_012500.mp4",
        sec = 0.9,
        objDetectModel = "model/frozen_inference_graph.pb",
        siameseModel = "model/Siamese_tracking.ckpt"
    )
    responce = stub.init(request)
    print(responce.status)

def runInference():
    channel = grpc.insecure_channel('localhost:50051')
    stub = visualNavigationMod_pb2_grpc.visualNavigationModserviceStub(channel)
    request = visualNavigationMod_pb2.inferenceArg(
        input = "input/Cut2-089-FY_01_20190825_082000.mp4",
        output = "output/test1.mp4"
    )
    responce = stub.inference(request)
    print(responce.score)

runInit()
runInference()
