import transformers
from transformers.pipelines import PIPELINE_REGISTRY

print(f"Transformers version: {transformers.__version__}")
print("\nAvailable pipeline tasks:")
# Newer versions use PIPELINE_REGISTRY.get_supported_tasks()
if hasattr(PIPELINE_REGISTRY, "get_supported_tasks"):
    print(sorted(PIPELINE_REGISTRY.get_supported_tasks()))
else:
    # Older versions might use SUPPORTED_TASKS dictionary
    try:
        # Attempt access via older potential path if needed
        from transformers.pipelines import SUPPORTED_TASKS
        print(sorted(list(SUPPORTED_TASKS.keys())))
    except ImportError:
        print("Could not determine supported tasks (check transformers installation).")