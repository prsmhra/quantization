import torch

#get quantized tensor by quantizing unquantized float tensor
float_tensor = torch.randn(2,2,3)
print(f"flaot tensor : {float_tensor}")
#Tensor quantization
scale, zero_point = 1e-4, 2
dtype = torch.qint32
q_per_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)
print(f"Quantized Tensor: {q_per_tensor}")

#Channel Quantization
scales = torch.tensor([1e-1,1e-2,1e-3])
zero_points = torch.tensor([-1,0,1])
channel_axis = 2
q_per_channel = torch.quantize_per_channel(float_tensor, scales, zero_points, axis=channel_axis, dtype=dtype)
print(f"Channel Quantized Tensor: {q_per_channel}")

#Create a quantized tensor directly from empty_quantized funtions
q = torch._empty_affine_quantized([10], scale=scale, zero_point=zero_point, dtype=dtype)
print(f"quentized tensor from empty quantized funtion;{q}")

#create a quantized tensor by assembling int tensor and quantization parameters
int_tensor = torch.randint(0, 100, size=(10,), dtype=torch.uint8)
# - torch.uint8 -> torch.quint8
# - torch.int8 -> torch.qint8
# - torch.int32 -> torch.qint32
q1 = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point) # no `dtype`
print(f"Quantized tensor by essembling int tensor and quantized parameters:{q1}")

#Dequantized

dequantized_tensor = q1.dequantize()
print(f"Dequantized Tensor: {dequantized_tensor} ")

#slicing 
s = q[2]
print(f"sliced qtensor: {s}")

#assignment

q[0]=3.7
print(q)

#copying
q2 = torch._empty_affine_quantized([10], scale=scale, zero_point=zero_point, dtype=dtype)
q2.copy_(q)
print(f"q2:{q2}")

#permutation
q3 = torch._empty_affine_quantized([2,3], scale=scale, zero_point=zero_point, dtype=dtype)
print(f"transpose:{q3.transpose(0,1)}")
print(f"permute:{q3.permute([1,0])}")
print(f"contiguous: {q2.contiguous()}")


# Serialization and Deserialization
import tempfile
with tempfile.NamedTemporaryFile() as f:
    torch.save(q2, f)
    f.seek(0)
    q4 = torch.load(f)
    print(f"q4:{q4}")
