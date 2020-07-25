
model_path = "/gdrive/My Drive/Adagrad/densenet_new.onnx"
input_size = 32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def build_engine(model_path):
#     with trt.Builder(TRT_LOGGER) as builder, \
#         builder.create_network() as network, \
#         trt.OnnxParser(network, TRT_LOGGER) as parser: 
#         builder.max_workspace_size = 1<<20
#         builder.max_batch_size = 1
#         print("here da...")
#         with open(model_path, "rb") as f:
#             parser.parse(f.read())
#         engine = builder.build_cuda_engine(network)
#         return engine

def get_engine(model_path):
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 10
        builder.fp16_mode = True if mode == 'fp16' else False
        builder.int8_mode = True if mode == 'int8' else False
        builder.max_workspace_size = 1 << 32  # 1GB:30

        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        print(len(network))  # Printed output == 0. Something is wrong. 

        engine = builder.build_cuda_engine(network)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
    return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

if __name__ == "__main__":
    inputs = np.random.random((10, 3, input_size, input_size)).astype(np.float32)
    # print(inputs)
    engine = get_engine(model_path)
    context = engine.create_execution_context()
    for _ in range(10):
        t1 = time.time()
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print(res)
        print("cost time: ", time.time()-t1)

