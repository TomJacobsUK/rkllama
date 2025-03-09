import ctypes
import os
import psutil
import re
from .classes import *
from .callback import *

# Connecter la fonction de rappel entre le côté Python et le côté C++
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Définir la classe RKLLM, qui inclut l'initialisation, l'inférence et les opérations de libération pour le modèle RKLLM dans la bibliothèque dynamique
class RKLLM(object):
    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None, max_context_len=None):
        
        self.format_schema = None
        self.format_type = None
        self.format_options = {}
        
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        # Estimate max context length if not specified
        if max_context_len is None:
            max_context_len = self.estimate_max_context_length(model_path)
            
        rkllm_param.max_context_len = max_context_len
        rkllm_param.max_new_tokens = -1
        rkllm_param.skip_special_token = True

        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.lora_adapter_path = None
        self.lora_model_name = None
        if lora_model_path:
            self.lora_adapter_path = lora_model_path
            self.lora_adapter_name = "test"

            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((self.lora_adapter_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((self.lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
        
        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def estimate_max_context_length(self, model_path):
        """
        Estimate the maximum context length based on model size and available memory.
        Optimized for embedded systems like RK3588 with limited RAM.
        
        Returns:
            int: Estimated maximum context length
        """
        # Get available system memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        
        print(f"System memory: {total_memory_gb:.2f} GB total, {available_memory_gb:.2f} GB available")
        
        # Extract model type/size from path if possible
        model_name = os.path.basename(model_path).lower()
        
        # Try to extract model size using regex (e.g., 7b, 3b, 1.5b, etc.)
        model_size_match = re.search(r'(\d+(\.\d+)?)[bB]', model_name)
        model_size_gb = None
        
        if model_size_match:
            size_in_b = float(model_size_match.group(1))
            # Rough estimation of model size in GB based on parameters
            # ~2 bytes per parameter for quantized models (4 bytes for FP32)
            model_size_gb = size_in_b * 2 / 1.0  # Assuming some level of quantization
        
        # Default values - more conservative for embedded devices
        token_memory_factor = 12  # bytes per token per position (initial conservative estimate)
        safety_factor = 0.6  # Lower safety factor for embedded systems
        
        # Adjust based on detected model size or use defaults
        if model_size_match:
            size_in_b = float(model_size_match.group(1))
            
            # Adjust parameters based on model size
            if size_in_b <= 1:
                base_memory = 2  # GB (approximate model size for 1B models)
                token_memory_factor = 8
            elif size_in_b <= 2:
                base_memory = 4  # GB
                token_memory_factor = 10
            elif size_in_b <= 3:
                base_memory = 6  # GB 
                token_memory_factor = 12
            elif size_in_b <= 7:
                base_memory = 14  # GB
            elif size_in_b <= 13:
                base_memory = 26  # GB
                token_memory_factor = 18
            elif size_in_b <= 34:
                base_memory = 68  # GB
                token_memory_factor = 24
                safety_factor = 0.5
            else:
                base_memory = 140  # GB
                token_memory_factor = 32
                safety_factor = 0.5
        else:
            # Default conservative guess for unknown model
            base_memory = 8  # GB
        
        is_memory_constrained = (total_memory_gb <= 8)
        
        if is_memory_constrained:
            print("Detected memory-constrained system (like RK3588)")
            # More conservative settings for constrained environments
            safety_factor = max(0.4, safety_factor - 0.1)
            # Increase memory efficiency assumption for systems with quantized models
            token_memory_factor = max(6, token_memory_factor - 2)
            
        # Calculate usable memory
        usable_memory_gb = max(0, available_memory_gb - (base_memory * 0.5)) * safety_factor
        
        # Convert to tokens (multiply by 1024^3 to convert GB to bytes)
        estimated_max_tokens = int((usable_memory_gb * 1024 * 1024 * 1024) / token_memory_factor)
        
        # Cap at reasonable values - lower cap for memory-constrained systems
        max_cap = 8192 if is_memory_constrained else 16384
        estimated_max_tokens = max(512, min(max_cap, estimated_max_tokens))
        
        # Round to nearest 512 for more granular control on constrained systems
        estimated_max_tokens = (estimated_max_tokens // 512) * 512
        
        print(f"Model: {model_name}")
        if model_size_gb:
            print(f"Estimated model size: ~{model_size_gb:.1f} GB")
        print(f"Estimated maximum context length: {estimated_max_tokens} tokens")
        
        return estimated_max_tokens

    def tokens_to_ctypes_array(self, tokens, ctype):
        return (ctype * len(tokens))(*tokens)

    def run(self, prompt_tokens):
        rkllm_lora_params = None
        if self.lora_model_name:
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((self.lora_model_name).encode('utf-8'))
        
        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = ctypes.byref(rkllm_lora_params) if rkllm_lora_params else None

        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_TOKEN

        if prompt_tokens[-1] != 2:  
            prompt_tokens.append(2)

        token_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)

        rkllm_input.input_data.token_input.input_ids = token_array
        rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(prompt_tokens))


        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)

        return

    def release(self):
        self.rkllm_destroy(self.handle)
