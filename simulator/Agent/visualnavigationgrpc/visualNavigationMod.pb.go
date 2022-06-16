// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.26.0
// 	protoc        v3.15.8
// source: visualNavigationMod.proto

package visualnavigationgrpc

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type InferenceArg struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Input  string `protobuf:"bytes,1,opt,name=input,proto3" json:"input,omitempty"`
	Output string `protobuf:"bytes,2,opt,name=output,proto3" json:"output,omitempty"`
}

func (x *InferenceArg) Reset() {
	*x = InferenceArg{}
	if protoimpl.UnsafeEnabled {
		mi := &file_visualNavigationMod_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *InferenceArg) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InferenceArg) ProtoMessage() {}

func (x *InferenceArg) ProtoReflect() protoreflect.Message {
	mi := &file_visualNavigationMod_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InferenceArg.ProtoReflect.Descriptor instead.
func (*InferenceArg) Descriptor() ([]byte, []int) {
	return file_visualNavigationMod_proto_rawDescGZIP(), []int{0}
}

func (x *InferenceArg) GetInput() string {
	if x != nil {
		return x.Input
	}
	return ""
}

func (x *InferenceArg) GetOutput() string {
	if x != nil {
		return x.Output
	}
	return ""
}

type InitArg struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Input          string  `protobuf:"bytes,1,opt,name=input,proto3" json:"input,omitempty"`
	Sec            float32 `protobuf:"fixed32,2,opt,name=sec,proto3" json:"sec,omitempty"`
	ObjDetectModel string  `protobuf:"bytes,3,opt,name=objDetectModel,proto3" json:"objDetectModel,omitempty"`
	SiameseModel   string  `protobuf:"bytes,4,opt,name=siameseModel,proto3" json:"siameseModel,omitempty"`
}

func (x *InitArg) Reset() {
	*x = InitArg{}
	if protoimpl.UnsafeEnabled {
		mi := &file_visualNavigationMod_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *InitArg) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InitArg) ProtoMessage() {}

func (x *InitArg) ProtoReflect() protoreflect.Message {
	mi := &file_visualNavigationMod_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InitArg.ProtoReflect.Descriptor instead.
func (*InitArg) Descriptor() ([]byte, []int) {
	return file_visualNavigationMod_proto_rawDescGZIP(), []int{1}
}

func (x *InitArg) GetInput() string {
	if x != nil {
		return x.Input
	}
	return ""
}

func (x *InitArg) GetSec() float32 {
	if x != nil {
		return x.Sec
	}
	return 0
}

func (x *InitArg) GetObjDetectModel() string {
	if x != nil {
		return x.ObjDetectModel
	}
	return ""
}

func (x *InitArg) GetSiameseModel() string {
	if x != nil {
		return x.SiameseModel
	}
	return ""
}

type Scores struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Score []float32 `protobuf:"fixed32,1,rep,packed,name=score,proto3" json:"score,omitempty"`
}

func (x *Scores) Reset() {
	*x = Scores{}
	if protoimpl.UnsafeEnabled {
		mi := &file_visualNavigationMod_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Scores) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Scores) ProtoMessage() {}

func (x *Scores) ProtoReflect() protoreflect.Message {
	mi := &file_visualNavigationMod_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Scores.ProtoReflect.Descriptor instead.
func (*Scores) Descriptor() ([]byte, []int) {
	return file_visualNavigationMod_proto_rawDescGZIP(), []int{2}
}

func (x *Scores) GetScore() []float32 {
	if x != nil {
		return x.Score
	}
	return nil
}

type InitOutput struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Status string `protobuf:"bytes,1,opt,name=status,proto3" json:"status,omitempty"`
}

func (x *InitOutput) Reset() {
	*x = InitOutput{}
	if protoimpl.UnsafeEnabled {
		mi := &file_visualNavigationMod_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *InitOutput) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InitOutput) ProtoMessage() {}

func (x *InitOutput) ProtoReflect() protoreflect.Message {
	mi := &file_visualNavigationMod_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InitOutput.ProtoReflect.Descriptor instead.
func (*InitOutput) Descriptor() ([]byte, []int) {
	return file_visualNavigationMod_proto_rawDescGZIP(), []int{3}
}

func (x *InitOutput) GetStatus() string {
	if x != nil {
		return x.Status
	}
	return ""
}

type InferenceOutput struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Status string  `protobuf:"bytes,1,opt,name=status,proto3" json:"status,omitempty"`
	Score  float32 `protobuf:"fixed32,2,opt,name=score,proto3" json:"score,omitempty"`
}

func (x *InferenceOutput) Reset() {
	*x = InferenceOutput{}
	if protoimpl.UnsafeEnabled {
		mi := &file_visualNavigationMod_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *InferenceOutput) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*InferenceOutput) ProtoMessage() {}

func (x *InferenceOutput) ProtoReflect() protoreflect.Message {
	mi := &file_visualNavigationMod_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use InferenceOutput.ProtoReflect.Descriptor instead.
func (*InferenceOutput) Descriptor() ([]byte, []int) {
	return file_visualNavigationMod_proto_rawDescGZIP(), []int{4}
}

func (x *InferenceOutput) GetStatus() string {
	if x != nil {
		return x.Status
	}
	return ""
}

func (x *InferenceOutput) GetScore() float32 {
	if x != nil {
		return x.Score
	}
	return 0
}

var File_visualNavigationMod_proto protoreflect.FileDescriptor

var file_visualNavigationMod_proto_rawDesc = []byte{
	0x0a, 0x19, 0x76, 0x69, 0x73, 0x75, 0x61, 0x6c, 0x4e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x4d, 0x6f, 0x64, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x14, 0x76, 0x69, 0x73,
	0x75, 0x61, 0x6c, 0x6e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x67, 0x72, 0x70,
	0x63, 0x22, 0x3c, 0x0a, 0x0c, 0x69, 0x6e, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x41, 0x72,
	0x67, 0x12, 0x14, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x16, 0x0a, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75,
	0x74, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x22,
	0x7d, 0x0a, 0x07, 0x69, 0x6e, 0x69, 0x74, 0x41, 0x72, 0x67, 0x12, 0x14, 0x0a, 0x05, 0x69, 0x6e,
	0x70, 0x75, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74,
	0x12, 0x10, 0x0a, 0x03, 0x73, 0x65, 0x63, 0x18, 0x02, 0x20, 0x01, 0x28, 0x02, 0x52, 0x03, 0x73,
	0x65, 0x63, 0x12, 0x26, 0x0a, 0x0e, 0x6f, 0x62, 0x6a, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74, 0x4d,
	0x6f, 0x64, 0x65, 0x6c, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0e, 0x6f, 0x62, 0x6a, 0x44,
	0x65, 0x74, 0x65, 0x63, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x12, 0x22, 0x0a, 0x0c, 0x73, 0x69,
	0x61, 0x6d, 0x65, 0x73, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x0c, 0x73, 0x69, 0x61, 0x6d, 0x65, 0x73, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x22, 0x1e,
	0x0a, 0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x63, 0x6f, 0x72,
	0x65, 0x18, 0x01, 0x20, 0x03, 0x28, 0x02, 0x52, 0x05, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x22, 0x24,
	0x0a, 0x0a, 0x69, 0x6e, 0x69, 0x74, 0x4f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x16, 0x0a, 0x06,
	0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x73, 0x74,
	0x61, 0x74, 0x75, 0x73, 0x22, 0x3f, 0x0a, 0x0f, 0x69, 0x6e, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63,
	0x65, 0x4f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x16, 0x0a, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75,
	0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x12,
	0x14, 0x0a, 0x05, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x02, 0x52, 0x05,
	0x73, 0x63, 0x6f, 0x72, 0x65, 0x32, 0xbd, 0x01, 0x0a, 0x1a, 0x76, 0x69, 0x73, 0x75, 0x61, 0x6c,
	0x4e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x4d, 0x6f, 0x64, 0x73, 0x65, 0x72,
	0x76, 0x69, 0x63, 0x65, 0x12, 0x47, 0x0a, 0x04, 0x69, 0x6e, 0x69, 0x74, 0x12, 0x1d, 0x2e, 0x76,
	0x69, 0x73, 0x75, 0x61, 0x6c, 0x6e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x67,
	0x72, 0x70, 0x63, 0x2e, 0x69, 0x6e, 0x69, 0x74, 0x41, 0x72, 0x67, 0x1a, 0x20, 0x2e, 0x76, 0x69,
	0x73, 0x75, 0x61, 0x6c, 0x6e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x67, 0x72,
	0x70, 0x63, 0x2e, 0x69, 0x6e, 0x69, 0x74, 0x4f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x56, 0x0a,
	0x09, 0x69, 0x6e, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x12, 0x22, 0x2e, 0x76, 0x69, 0x73,
	0x75, 0x61, 0x6c, 0x6e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x67, 0x72, 0x70,
	0x63, 0x2e, 0x69, 0x6e, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x41, 0x72, 0x67, 0x1a, 0x25,
	0x2e, 0x76, 0x69, 0x73, 0x75, 0x61, 0x6c, 0x6e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x67, 0x72, 0x70, 0x63, 0x2e, 0x69, 0x6e, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63, 0x65, 0x4f,
	0x75, 0x74, 0x70, 0x75, 0x74, 0x42, 0x17, 0x5a, 0x15, 0x2f, 0x76, 0x69, 0x73, 0x75, 0x61, 0x6c,
	0x6e, 0x61, 0x76, 0x69, 0x67, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x67, 0x72, 0x70, 0x63, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_visualNavigationMod_proto_rawDescOnce sync.Once
	file_visualNavigationMod_proto_rawDescData = file_visualNavigationMod_proto_rawDesc
)

func file_visualNavigationMod_proto_rawDescGZIP() []byte {
	file_visualNavigationMod_proto_rawDescOnce.Do(func() {
		file_visualNavigationMod_proto_rawDescData = protoimpl.X.CompressGZIP(file_visualNavigationMod_proto_rawDescData)
	})
	return file_visualNavigationMod_proto_rawDescData
}

var file_visualNavigationMod_proto_msgTypes = make([]protoimpl.MessageInfo, 5)
var file_visualNavigationMod_proto_goTypes = []interface{}{
	(*InferenceArg)(nil),    // 0: visualnavigationgrpc.inferenceArg
	(*InitArg)(nil),         // 1: visualnavigationgrpc.initArg
	(*Scores)(nil),          // 2: visualnavigationgrpc.scores
	(*InitOutput)(nil),      // 3: visualnavigationgrpc.initOutput
	(*InferenceOutput)(nil), // 4: visualnavigationgrpc.inferenceOutput
}
var file_visualNavigationMod_proto_depIdxs = []int32{
	1, // 0: visualnavigationgrpc.visualNavigationModservice.init:input_type -> visualnavigationgrpc.initArg
	0, // 1: visualnavigationgrpc.visualNavigationModservice.inference:input_type -> visualnavigationgrpc.inferenceArg
	3, // 2: visualnavigationgrpc.visualNavigationModservice.init:output_type -> visualnavigationgrpc.initOutput
	4, // 3: visualnavigationgrpc.visualNavigationModservice.inference:output_type -> visualnavigationgrpc.inferenceOutput
	2, // [2:4] is the sub-list for method output_type
	0, // [0:2] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_visualNavigationMod_proto_init() }
func file_visualNavigationMod_proto_init() {
	if File_visualNavigationMod_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_visualNavigationMod_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*InferenceArg); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_visualNavigationMod_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*InitArg); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_visualNavigationMod_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Scores); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_visualNavigationMod_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*InitOutput); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_visualNavigationMod_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*InferenceOutput); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_visualNavigationMod_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   5,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_visualNavigationMod_proto_goTypes,
		DependencyIndexes: file_visualNavigationMod_proto_depIdxs,
		MessageInfos:      file_visualNavigationMod_proto_msgTypes,
	}.Build()
	File_visualNavigationMod_proto = out.File
	file_visualNavigationMod_proto_rawDesc = nil
	file_visualNavigationMod_proto_goTypes = nil
	file_visualNavigationMod_proto_depIdxs = nil
}