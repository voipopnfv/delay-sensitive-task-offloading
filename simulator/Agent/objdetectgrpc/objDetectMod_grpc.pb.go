// Code generated by protoc-gen-go-grpc. DO NOT EDIT.

package objdetectgrpc

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// ObjDetectModserviceClient is the client API for ObjDetectModservice service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type ObjDetectModserviceClient interface {
	Init(ctx context.Context, in *InitArg, opts ...grpc.CallOption) (*InitOutput, error)
	Inference(ctx context.Context, in *InferenceArg, opts ...grpc.CallOption) (*InferenceOutput, error)
}

type objDetectModserviceClient struct {
	cc grpc.ClientConnInterface
}

func NewObjDetectModserviceClient(cc grpc.ClientConnInterface) ObjDetectModserviceClient {
	return &objDetectModserviceClient{cc}
}

func (c *objDetectModserviceClient) Init(ctx context.Context, in *InitArg, opts ...grpc.CallOption) (*InitOutput, error) {
	out := new(InitOutput)
	err := c.cc.Invoke(ctx, "/objdetectgrpc.objDetectModservice/init", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *objDetectModserviceClient) Inference(ctx context.Context, in *InferenceArg, opts ...grpc.CallOption) (*InferenceOutput, error) {
	out := new(InferenceOutput)
	err := c.cc.Invoke(ctx, "/objdetectgrpc.objDetectModservice/inference", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ObjDetectModserviceServer is the server API for ObjDetectModservice service.
// All implementations must embed UnimplementedObjDetectModserviceServer
// for forward compatibility
type ObjDetectModserviceServer interface {
	Init(context.Context, *InitArg) (*InitOutput, error)
	Inference(context.Context, *InferenceArg) (*InferenceOutput, error)
	mustEmbedUnimplementedObjDetectModserviceServer()
}

// UnimplementedObjDetectModserviceServer must be embedded to have forward compatible implementations.
type UnimplementedObjDetectModserviceServer struct {
}

func (UnimplementedObjDetectModserviceServer) Init(context.Context, *InitArg) (*InitOutput, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Init not implemented")
}
func (UnimplementedObjDetectModserviceServer) Inference(context.Context, *InferenceArg) (*InferenceOutput, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Inference not implemented")
}
func (UnimplementedObjDetectModserviceServer) mustEmbedUnimplementedObjDetectModserviceServer() {}

// UnsafeObjDetectModserviceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ObjDetectModserviceServer will
// result in compilation errors.
type UnsafeObjDetectModserviceServer interface {
	mustEmbedUnimplementedObjDetectModserviceServer()
}

func RegisterObjDetectModserviceServer(s grpc.ServiceRegistrar, srv ObjDetectModserviceServer) {
	s.RegisterService(&ObjDetectModservice_ServiceDesc, srv)
}

func _ObjDetectModservice_Init_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(InitArg)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ObjDetectModserviceServer).Init(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/objdetectgrpc.objDetectModservice/init",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ObjDetectModserviceServer).Init(ctx, req.(*InitArg))
	}
	return interceptor(ctx, in, info, handler)
}

func _ObjDetectModservice_Inference_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(InferenceArg)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ObjDetectModserviceServer).Inference(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/objdetectgrpc.objDetectModservice/inference",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ObjDetectModserviceServer).Inference(ctx, req.(*InferenceArg))
	}
	return interceptor(ctx, in, info, handler)
}

// ObjDetectModservice_ServiceDesc is the grpc.ServiceDesc for ObjDetectModservice service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var ObjDetectModservice_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "objdetectgrpc.objDetectModservice",
	HandlerType: (*ObjDetectModserviceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "init",
			Handler:    _ObjDetectModservice_Init_Handler,
		},
		{
			MethodName: "inference",
			Handler:    _ObjDetectModservice_Inference_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "objDetectMod.proto",
}
