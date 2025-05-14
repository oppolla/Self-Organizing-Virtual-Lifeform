from setuptools import setup

setup(
    name="sovl-system",
    version="0.1.1",
    description="Self-Organizing Virtual Lifeform (SOVL) AI agent framework",
    author="Your Name",
    py_modules=[
        "run_sovl",
        "sovl_config", "sovl_logger", "sovl_resource", "sovl_state", "sovl_error", "sovl_memory", "sovl_events", "sovl_io", "sovl_queue", "sovl_hardware", "sovl_data", "sovl_manager", "sovl_processor", "sovl_engram", "sovl_scaffold", "sovl_monitor", "sovl_main", "sovl_interfaces", "sovl_generation", "sovl_confidence", "sovl_bonder", "sovl_temperament", "sovl_curiosity", "sovl_viber", "sovl_dreamer", "sovl_striver", "sovl_meditater", "sovl_shamer", "sovl_trainer", "sovl_primer", "sovl_api", "sovl_cli", "sovl_recaller"
    ],
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "pydantic",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "sovl=run_sovl:main",
        ],
    },
    include_package_data=True,
    package_data={"": ["sovl_config.json", "sovl_config.defaults.json"]},
    python_requires=">=3.8",
)
