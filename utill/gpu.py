import os
import tensorflow as tf

def setup_gpu():
    """
    Configure GPU settings for optimal usage.
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPU devices available. Running on CPU.")
        return False
    
    try:
        # Configure GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set visible devices to first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        print(f"Number of GPUs available: {len(gpus)}")
        print(f"Using GPU: {gpus[0].name}")
        
        # Set environment variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
        
        return True
    
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        return False

def get_gpu_memory_info():
    """
    Get GPU memory information
    Returns:
        str: GPU memory usage information
    """
    try:
        # Get GPU device
        gpu = tf.config.list_physical_devices('GPU')[0]
        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
        
        return f"GPU Memory Usage - Peak: {gpu_memory['peak']/1e6:.2f}MB, Current: {gpu_memory['current']/1e6:.2f}MB"
    except:
        return "GPU memory information not available"